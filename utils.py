import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import SimpleITK as sitk
import numpy as np
from PIL import Image
import shutil
from config import Config as cfg
from scipy import ndimage as nd
import os
import cv2
import random
import math
from sklearn.cluster import KMeans
from sklearn import manifold
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

import pystrum.pynd.ndutils as nd
import torchvision.utils as vutils
def jacobian_determinant(disp):
    """
    jacobian determinant of a displacement field.
    NB: to compute the spatial gradients, we use np.gradient.
    Parameters:
        disp: 2D or 3D displacement field of size [*vol_shape, nb_dims],
              where vol_shape is of len nb_dims
    Returns:
        jacobian determinant (scalar)
    """

    # check inputs
    volshape = disp.shape[:-1]
    nb_dims = len(volshape)
    assert len(volshape) in (2, 3), 'flow has to be 2D or 3D'

    # compute grid
    grid_lst = nd.volsize2ndgrid(volshape)
    grid = np.stack(grid_lst, len(volshape))

    # compute gradients
    J = np.gradient(disp + grid)

    # 3D glow
    if nb_dims == 3:
        dx = J[0]
        dy = J[1]
        dz = J[2]

        # compute jacobian components
        Jdet0 = dx[..., 0] * (dy[..., 1] * dz[..., 2] - dy[..., 2] * dz[..., 1])
        Jdet1 = dx[..., 1] * (dy[..., 0] * dz[..., 2] - dy[..., 2] * dz[..., 0])
        Jdet2 = dx[..., 2] * (dy[..., 0] * dz[..., 1] - dy[..., 1] * dz[..., 0])

        return Jdet0 - Jdet1 + Jdet2

    else:  # must be 2

        dfdx = J[0]
        dfdy = J[1]

        return dfdx[..., 0] * dfdy[..., 1] - dfdy[..., 0] * dfdx[..., 1]


def dice_(array1, array2, labels):
    """
    Computes the dice overlap between two arrays for a given set of integer labels.
    """
    array1 = array1.reshape(1, -1)
    array2 = array2.reshape(1, -1)
    dicem = np.zeros(len(labels))
    for idx, label in enumerate(labels):
        top = 2 * np.sum(np.logical_and(array1 == label, array2 == label))
        bottom = np.sum(array1 == label) + np.sum(array2 == label)
        bottom = np.maximum(bottom, np.finfo(float).eps)  # add epsilon
        dicem[idx] = top / bottom
    return dicem

def dice(vol1, vol2, labels=None, nargout=1):
    '''
    Dice [1] volume overlap metric

    The default is to *not* return a measure for the background layer (label = 0)

    [1] Dice, Lee R. "Measures of the amount of ecologic association between species."
    Ecology 26.3 (1945): 297-302.

    Parameters
    ----------
    vol1 : nd array. The first volume (e.g. predicted volume)
    vol2 : nd array. The second volume (e.g. "true" volume)
    labels : optional vector of labels on which to compute Dice.
        If this is not provided, Dice is computed on all non-background (non-0) labels
    nargout : optional control of output arguments. if 1, output Dice measure(s).
        if 2, output tuple of (Dice, labels)

    Output
    ------
    if nargout == 1 : dice : vector of dice measures for each labels
    if nargout == 2 : (dice, labels) : where labels is a vector of the labels on which
        dice was computed
    '''
    if labels is None:
        labels = np.unique(np.concatenate((vol1, vol2)))
        labels = np.delete(labels, np.where(labels == 0))  # remove background

    dicem = np.zeros(len(labels))
    for idx, lab in enumerate(labels):
        vol1l = vol1 == lab
        vol2l = vol2 == lab
        top = 2 * np.sum(np.logical_and(vol1l, vol2l))
        bottom = np.sum(vol1l) + np.sum(vol2l)
        bottom = np.maximum(bottom, np.finfo(float).eps)  # add epsilon.
        dicem[idx] = top / bottom

    if nargout == 1:
        return dicem
    else:
        return (dicem, labels)


def calculate_discount(r, bootstrap):
    size = len(r)
    R_batch = np.zeros(size, np.float32)
    R = bootstrap
    for i in reversed(range(0, size)):
        R = r[i] + cfg.GAMMA * R
        R_batch[i] = R

    return R_batch


def calculate_advantage(r, v, bootstrap):
    """
    Calulates the Advantage Funtion for each timestep t in the batch, where:
        - At = \sum_{i=0}^{k-1} gamma^{i} * r_{t+i} + gamma^{k} * V(s_{t+k}) - V(s_{t})
    where V(s_{t+k}) is the bootstraped value (if the episode hasn't finished).
    This results in a 1-step update for timestep B, 2-step update for timestep
    B-1,...., B-step discounted reward for timestep 1, where:
        - B = Batch Size
    Example: consider B = 3. Therefore, it results in the following advantage vector:
        - A[0] = r_1 + gamma * r_2 + gamma^2 * r_3 + gamma^3 * bootstrap - V(s_1)
        - A[1] = r_2 + gamma * r_3 + gamma^2 * bootstrap - V(s_2)
        - A[2] = r_3 + gamma * bootstrap - V(s_3)
    """
    size = len(r)
    A_batch = np.zeros([size], np.float32)
    aux = bootstrap
    for i in reversed(range(0, size)):
        aux = r[i] + cfg.GAMMA * aux
        A = aux - v[i]
        A_batch[i] = A

    return A_batch


def mkdir(path):
    # give a path, create the folder
    folder = os.path.exists(path)

    if not folder:
        os.makedirs(path)

def deldir(path):
    folder = os.path.exists(path)
    if folder:
        shutil.rmtree(path)

def remkdir(path):
    deldir(path)
    mkdir(path)

def load_image(filename):
    im = Image.open(filename)
    return im.convert('L')

def resize(img, size):
    # return cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    or_size = img.shape
    zoom_factor = [s/o for o, s in zip(or_size, size)]
    return nd.interpolation.zoom(img, zoom=zoom_factor)

def read_data(root):
    ct_list = os.listdir(root+'/CT')
    mr_list = os.listdir(root+'/MR')

    ct_list.sort()
    mr_list.sort()

    datas = [(root+'/CT/'+ct, root+'/MR/'+mr) for ct, mr in zip(ct_list, mr_list)]
    return datas

def segmentation(img):
    shape = img.shape
    img = img.reshape(-1, 1)
    model = KMeans(n_clusters=3)
    labels = model.fit_predict(img)
    labels = labels.reshape(shape)
    # print(labels.shape)
    # cv2.imwrite('process/labels.png', labels[:, 64, :])
    return labels

def virtual_label(size=168, num=30):
    if isinstance(size, int):
        size = (size, size)
    label_im = np.zeros(size).astype(np.uint8)
    xs = np.linspace(0, size[0], num, False, dtype=np.int)
    # xs = np.hstack((xs, xs+1))
    ys = np.linspace(0, size[1], num, False, dtype=np.int)
    # ys = np.hstack((ys, ys+1))
    x, y = np.meshgrid(xs, ys)

    label_im[x.ravel(), y.ravel()] = 255
    # im = Image.fromarray(label_im).convert('1')
    # for i, xi in enumerate(set(x.ravel())):
    #     label_im[xi, y.ravel()] = i*10
    return label_im

def perturb(im, mesh_size=(5, 5, 5), tmp_level=12):
    sitk_im = sitk.GetImageFromArray(im)
    bspline = sitk.BSplineTransform(3, 3) # dimension, spline_order
    bspline.SetTransformDomainOrigin([0., 0., 0.]) # origin
    bspline.SetTransformDomainDirection([1., 0., 0. , 0., 1., 0., 0., 0., 1.]) # direction_matrix_row_major
    bspline.SetTransformDomainPhysicalDimensions(sitk_im.GetSize())
    bspline.SetTransformDomainMeshSize(mesh_size)

    originalControlPointDisplacements = np.random.randn(len(bspline.GetParameters()))*tmp_level
    bspline.SetParameters(originalControlPointDisplacements)
    # ###################### bspline to displacement #################
    # transform_to_displacement_filter = sitk.TransformToDisplacementFieldFilter()
    # transform_to_displacement_filter.SetReferenceImage(ground_truth)
    # displacement = transform_to_displacement_filter.Execute(bspline)

    new_im = sitk.Resample(sitk_im, bspline)

    return sitk.GetArrayFromImage(new_im)


def displacement_score(predict, target):
    """
        predict: predicted displacement field
        target: target displacement field

        return: mse score of predicted displacement field
    """
    return np.power(predict - target, 2).mean()


def tensor_im(data):
    im = Image.fromarray(data).convert('L')
    return TF.to_tensor(im)

def tensor(data, dtype=torch.float32, device=None):
    return torch.as_tensor(data, dtype=dtype, device=device)

def numpy_im(data, scale=255, device=None):
    # 如果输入是PyTorch Tensor，则转换为NumPy数组
    if isinstance(data, torch.Tensor):
        # data = data.cpu().detach().numpy() if device is not None else data.detach().numpy()
        data = numpy(data, device)
    # 直接处理NumPy数组
    im = np.squeeze(data) * scale
    # im = numpy(data, device).squeeze() * scale
    return im.astype(np.uint8)

def numpy_im_round(data, scale=255, device=None):
    # 如果输入是PyTorch Tensor，则转换为NumPy数组
    if isinstance(data, torch.Tensor):
        # data = data.cpu().detach().numpy() if device is not None else data.detach().numpy()
        data = numpy(data, device)
    # 直接处理NumPy数组
    im = np.squeeze(data) * scale
    # im = numpy(data, device).squeeze() * scale
    return np.round(im).astype(np.uint8)

def numpy(data, device=None):
    if device is not None:
        return data.detach().cpu().numpy()
    else:
        return data.detach().numpy()

def render_flow(flow, coef = 15, thresh = 250):

    im_flow = np.zeros((3, cfg.HEIGHT, cfg.WIDTH))
    im_flow[:] = flow
    im_flow = im_flow.transpose(1, 2, 0)
    #im_flow = 0.5 + im_flow / coef
    im_flow = np.abs(im_flow)
    im_flow = np.exp(-im_flow / coef)
    im_flow = im_flow * thresh
    #im_flow = 1 - im_flow / 20
    return im_flow

def visualize(fixed_seg, pred_seg, threshold=80):
    img = np.zeros((cfg.HEIGHT, cfg.WIDTH, 3))
    img[..., 1] = (pred_seg)
    img[..., 2] =  (fixed_seg)

    return (img>threshold)*255


def make_seg_with_labels(seg, labels):
    new_seg = np.zeros_like(seg)
    for label in labels:
        new_seg[seg==label] = label

    return new_seg


# def render_image_with_mask(img, mask, color=0, colorful=False):
#     pink= [255,105,180]
#     yellow = [252, 252, 27]
#     green = [0,255,0]
#     gold = [255,215,0]
#     colors = [pink, yellow,gold]
#     # mask = cv2.blur(mask, (3, 1))
#
#     img = sitk.GetImageFromArray(img)
#     mask = sitk.GetImageFromArray(mask)
#     if colorful:
#         im = sitk.LabelOverlay(img, sitk.ConnectedComponent(mask))
#     else:
#         im = sitk.LabelOverlay(img, mask, colormap=colors[color])
#     return sitk.GetArrayFromImage(im)


def render_image_with_mask(img, mask, color=0, colorful=False):
    pink= [255,105,180]
    yellow = [252, 252, 27]
    green = [0,255,0]
    gold = [255,215,0]
    colors = [pink, yellow, gold]
    # mask = cv2.blur(mask, (3, 1))
    # mask = mask.cpu().numpy()

    img = sitk.GetImageFromArray(img)
    mask = sitk.GetImageFromArray(mask)
    img = sitk.Cast(img, sitk.sitkUInt16)
    mask = sitk.Cast(mask, sitk.sitkUInt16)
    if colorful:
        # im = sitk.LabelOverlay(img, sitk.ConnectedComponent(mask))
        im = sitk.LabelOverlay(img, mask)
    else:
        im = sitk.LabelOverlay(img, mask, colormap=colors[color])
    return sitk.GetArrayFromImage(im)

# 专门针对心脏
def plot_label(img, mask, color=0, colorful=False, path=None):
    z_middle = img.shape[2] // 2
    # img = img[:, :, z_middle]
    # mask = mask[:, :, z_middle]
    pink = [255, 105, 180]
    yellow = [252, 252, 27]
    green = [0, 255, 0]
    gold = [255, 215, 0]
    colors = [pink, yellow, gold]

    img = sitk.GetImageFromArray(img)
    mask = sitk.GetImageFromArray(mask)
    img = sitk.Cast(img, sitk.sitkUInt16)
    mask = sitk.Cast(mask, sitk.sitkUInt16)
    if colorful:
        im = sitk.LabelOverlay(img, mask, opacity=0.5)
    else:
        im = sitk.LabelOverlay(img, mask, opacity=0.5, colormap=colors[color])
    img = sitk.GetArrayFromImage(im)

    # 无背景图
    # cv2.imwrite(path, img[:, :, z_middle, :])
    # img_tensor = torch.tensor(img.astype(np.float32))
    # vutils.save_image(img_tensor[:, :, z_middle, :].data, path, normalize=True)
    # 有背景图
    plt.plot()
    plt.imshow(img[:, :, z_middle, :])
    plt.axis('off')
    plt.savefig(path, bbox_inches='tight', pad_inches=0)

    plt.clf()


# Show dataset images with T-sne projection of latent space encoding
def computeTSNEProjectionOfLatentSpace(latent, labs, display=True, idx=0, step=0):
    # Compute latent space representation
    print("Computing latent space projection...")
    # dist, _ = encoder(X)
    # X_encoded = dist.sample().cpu().numpy()
    # print(X_encoded.shape)

    # Compute t-SNE embedding of latent space
    print("Computing t-SNE embedding...")
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    X_tsne = tsne.fit_transform(latent)
    print(X_tsne.shape)
    # Plot images according to t-sne embedding
    if display:
        print("Plotting t-SNE visualization...")
        fig, ax = plt.subplots()
        # imscatter(X_tsne[:, 0], X_tsne[:, 1], imageData=X.squeeze().cpu().numpy(), ax=ax, zoom=0.6)
        ax.scatter(X_tsne[:,0],X_tsne[:,1], c=labs, s=4, cmap='tab10')
        plt.xticks([])
        plt.yticks([])
        mkdir('latent')
        plt.savefig(f'latent/{idx}_{step}.png', bbox_inches = 'tight')
        plt.close()
    else:
        return X_tsne

# Scatter with images instead of points
def imscatter(x, y, ax, imageData, zoom):
    images = []
    for i in range(len(x)):
        x0, y0 = x[i], y[i]
        # Convert to image
        img = imageData[i]*255.
        img = img.astype(np.uint8).reshape([-1, 128, 128, 128])
        img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
        # Note: OpenCV uses BGR and plt uses RGB
        image = OffsetImage(img, zoom=zoom)
        ab = AnnotationBbox(image, (x0, y0), xycoords='data', frameon=False)
        images.append(ax.add_artist(ab))

    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()


def add_weight_decay(net, l2_value, skip_list=()):
    decay, no_decay = [], []
    for name, param in net.named_parameters():
        if not param.requires_grad: continue # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list: no_decay.append(param)
        else: decay.append(param)
    return [{'params': no_decay, 'weight_decay': 0.}, {'params': decay, 'weight_decay': l2_value}]


def multi_scale_ncc_loss(I, J, win=9, scale=3):
    """
    local (over window) normalized cross correlation
    """

    total_NCC = 0
    for i in range(scale):
        current_ncc = ncc_loss(I, J, win=win - (i*2))
        total_NCC += current_ncc/(2**i)

        I = F.avg_pool3d(I, kernel_size=3, stride=2, padding=1, count_include_pad=False)
        J = F.avg_pool3d(J, kernel_size=3, stride=2, padding=1, count_include_pad=False)

    return total_NCC

def ncc_loss(I, J, win=None):
    return 1 - ncc_tensor_score(I, J, win=win)

def ncc_tensor_score(I, J, win=None):
    """
    calculate the normalize cross correlation between I and J
    assumes I, J are sized [batch_size, *vol_shape, nb_feats]
    """

    ndims = len(list(I.size())) - 2
    assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

    if win is None:
        win = [9] * ndims
    else:
        win = [win] * ndims

    conv_fn = getattr(F, 'conv%dd' % ndims)
    I2 = I*I
    J2 = J*J
    IJ = I*J

    sum_filt = torch.ones([1, 1, *win]).to("cuda")

    pad_no = math.floor(win[0]/2)

    if ndims == 1:
        stride = (1)
        padding = (pad_no)
    elif ndims == 2:
        stride = (1,1)
        padding = (pad_no, pad_no)
    else:
        stride = (1,1,1)
        padding = (pad_no, pad_no, pad_no)

    I_var, J_var, cross = compute_local_sums(I, J, sum_filt, stride, padding, win)

    cc = cross*cross / (I_var*J_var + 1e-5)

    return torch.mean(cc)

def compute_local_sums(I, J, filt, stride, padding, win):
    I2 = I * I
    J2 = J * J
    IJ = I * J

    I_sum = F.conv3d(I, filt, stride=stride, padding=padding)
    J_sum = F.conv3d(J, filt, stride=stride, padding=padding)
    I2_sum = F.conv3d(I2, filt, stride=stride, padding=padding)
    J2_sum = F.conv3d(J2, filt, stride=stride, padding=padding)
    IJ_sum = F.conv3d(IJ, filt, stride=stride, padding=padding)

    win_size = np.prod(win)
    u_I = I_sum / win_size
    u_J = J_sum / win_size

    cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
    I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
    J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

    return I_var, J_var, cross



def gradient_loss(s, penalty='l2'):
    dy = torch.abs(s[:, :, 1:, :, :] - s[:, :, :-1, :, :]) - 1e-2
    dx = torch.abs(s[:, :, :, 1:, :] - s[:, :, :, :-1, :]) - 1e-2
    dz = torch.abs(s[:, :, :, :, 1:] - s[:, :, :, :, :-1]) - 1e-2
    # print(dx.size(), dy.size())

    if(penalty == 'l2'):
        dy = dy * dy
        dx = dx * dx
        dz = dz * dz

    d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
    return d / 3.0

def dice_loss(pred, target):
    return -dice_score(pred, target)

def dice_score(pred, target):
    """This definition generalize to real valued pred and target vector.
This should be differentiable.
    pred: tensor with first dimension as batch
    target: tensor with first dimension as batch
    """
    top = 2 * torch.sum(pred * target, [1, 2, 3])
    union = torch.sum(pred + target, [1, 2, 3])
    eps = torch.ones_like(union) * 1e-10
    bottom = torch.max(union, eps)
    dice = torch.mean(top / bottom)
    #print("Dice score", dice)
    return dice

def convert_score(score):
    res = 1 if score >0 else -1
    return res


def aff_flow(W, b, xs, ys, zs):
    b = b.view(-1, 1, 1, 1, 3)
    xr = torch.arange(-(xs-1)/2., xs/2., 1.0, dtype=torch.float32).view(1, -1, 1, 1, 1)
    yr = torch.arange(-(ys-1)/2., ys/2., 1.0, dtype=torch.float32).view(1, 1, -1, 1, 1)
    zr = torch.arange(-(zs-1)/2., zs/2., 1.0, dtype=torch.float32).view(1, 1, 1, -1, 1)

    xr = xr.cuda()
    yr = yr.cuda()
    zr = zr.cuda()

    wx = W[:, :, 0].view(-1, 1, 1, 1, 3)
    wy = W[:, :, 1].view(-1, 1, 1, 1, 3)
    wz = W[:, :, 2].view(-1, 1, 1, 1, 3)

    flow = xr*wx + yr*wy + zr*wz + b

    return flow.permute(0, 4, 1, 2, 3)


def l2_det_loss(M):
    # M = [[M[:, i, j] for j in range(3)] for i in range(3)]
    # det = ((
    #        M[0][0] * M[1][1] * M[2][2]
    #      + M[0][1] * M[1][2] * M[2][0]
    #      + M[0][2] * M[1][0] * M[2][1]
    #     ) - (
    #        M[0][0] * M[1][2] * M[2][1]
    #      + M[0][1] * M[1][0] * M[2][2]
    #      + M[0][2] * M[1][1] * M[2][0]))
    det = torch.det(M)

    return torch.mean((det - 1.).pow(2))

def ortho_loss(A, eps=1e-5):
    I = torch.eye(3).cuda()
    epsI = I*eps
    C = torch.mul(A.transpose(2, 1), A) + epsI
    # print(C[0])
    s = torch.symeig(C, eigenvectors=True)[0]
    s1, s2, s3 = s[:, 0], s[:, 1], s[:, 2]

    def elem_sym_polys_of_eigen_values(M):
            M = [[M[:, i, j] for j in range(3)] for i in range(3)]
            sigma1 = M[0][0]+ M[1][1] + M[2][2]

            sigma2 = ((M[0][0] * M[1][1]
                     + M[1][1] * M[2][2]
                     + M[2][2] * M[0][0])
                     -
                      (M[0][1] * M[1][0]
                     + M[1][2] * M[2][1]
                     + M[2][0] * M[0][2]))

            sigma3 = ((M[0][0] * M[1][1] * M[2][2]
                     + M[0][1] * M[1][2] * M[2][0]
                     + M[0][2] * M[1][0] * M[2][1])
                     -
                      (M[0][0] * M[1][2] * M[2][1]
                     + M[0][1] * M[1][0] * M[2][2]
                     + M[0][2] * M[1][1] * M[2][0]))

            return sigma1, sigma2, sigma3

    # sigma1, sigma2, sigma3 represent a+b+c, ab+ac+bc, abc respectively
    # s1, s2, s3 = elem_sym_polys_of_eigen_values(C)
    # print(s1, s2, s3)

    # ortho_loss = s1 + (1 + eps) * (1 + eps) * s2 / s3 - 3 * 2 * (1 + eps)

    ortho_loss = (s1 + (1 + eps) * (1 + eps) / s1) \
               + (s2 + (1 + eps) * (1 + eps) / s2) \
               + (s3 + (1 + eps) * (1 + eps) / s3) - 2 * 3 * (1 + eps)

    return torch.mean(ortho_loss)


# --------------------- F-SAM -----------------------------------

class FriendlySAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, sigma=1, lmbda=0.9, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(FriendlySAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)
        self.sigma = sigma
        self.lmbda = lmbda
        print('FriendlySAM sigma:', self.sigma, 'lambda:', self.lmbda)

    @torch.no_grad()
    def first_step(self, zero_grad=False):

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                grad = p.grad.clone()
                if not "momentum" in self.state[p]:
                    self.state[p]["momentum"] = grad
                else:
                    p.grad -= self.state[p]["momentum"] * self.sigma
                    self.state[p]["momentum"] = self.state[p]["momentum"] * self.lmbda + grad * (1 - self.lmbda)

        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][
            0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
            torch.stack([
                ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups



def plot_flow_grid(flow, interval=3, path=None):
    disp_xy = flow[0, :2, :, :, :]  # 选择第一个和第二个通道，表示 x 和 y 方向的位移
    z_middle = disp_xy.shape[3] // 2
    disp_xy_slice = disp_xy[:, :, :, z_middle]
    disp_xy_slice = disp_xy_slice.cpu().numpy()
    background = np.zeros(disp_xy_slice.shape[1:])

    id_grid_H, id_grid_W = np.meshgrid(range(0, background.shape[0] - 1, interval),
                                       range(0, background.shape[1] - 1, interval),
                                       indexing='ij')
    new_grid_H = id_grid_H + disp_xy_slice[0, id_grid_H, id_grid_W]
    new_grid_W = id_grid_W + disp_xy_slice[1, id_grid_H, id_grid_W]

    kwargs = {"linewidth": 1.5, "color": 'c'}
    # matplotlib.plot() uses CV x-y indexing
    ax = plt.subplot(1, 1, 1)
    for i in range(new_grid_H.shape[0]):
        ax.plot(new_grid_W[i, :], new_grid_H[i, :], **kwargs)  # each draws a horizontal line
    for i in range(new_grid_H.shape[1]):
        ax.plot(new_grid_W[:, i], new_grid_H[:, i], **kwargs)  # each draws a vertical line

    ax.imshow(background, cmap='gray')
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    plt.savefig(path,bbox_inches='tight', pad_inches=0)
    plt.clf()

# def plot_difference(fixed_img, warped_img, path):
#     z_middle = fixed_img.shape[2] // 2
#     img_arr_diff = warped_img[:, :, z_middle] - fixed_img[:, :, z_middle]
#     # img_arr_diff = np.rot90(img_arr_diff, 1)
#     plt.plot()
#     plt.imshow(img_arr_diff, cmap='bwr')
#     plt.clim(-0.5, 0.5)
#     plt.xticks([])
#     plt.yticks([])
#     plt.colorbar(fraction=0.046, pad=0.04)
#     plt.savefig(path, bbox_inches='tight')
#     plt.clf()

def plot_difference(fixed_img, warped_img, path):
    # 选择中间切片
    z_middle = fixed_img.shape[2] // 2

    # 判断是否需要归一化的函数
    def normalize_if_needed(img):
        # 检查数据是否在 [0, 1] 范围内
        if img.min() < 0 or img.max() > 1:
            return img / 255.0  # 如果数据超出范围，归一化到 [0, 1]
        return img  # 如果数据已经在 [0, 1] 范围，直接返回

    # 判断是否需要归一化的函数 这个慎用吧 不同的项目进行不同处理就好
    # def normalize_if_needed(img):
    #     # 获取最大值和最小值
    #     img_min = np.min(img)
    #     img_max = np.max(img)
    #
    #     # 如果最大值不等于最小值（避免除以0）
    #     if img_max != img_min:
    #         # 归一化到 [0, 1] 范围
    #         return (img - img_min) / (img_max - img_min)
    #     else:
    #         # 如果图像值相同，直接返回
    #         return img

    # 对两张图像进行判断并归一化（如果需要）
    fixed_img_normalized = normalize_if_needed(fixed_img[:, :, z_middle])
    warped_img_normalized = normalize_if_needed(warped_img[:, :, z_middle])

    # 计算归一化后的差异
    img_arr_diff = warped_img_normalized - fixed_img_normalized

    # 显示差异图像，不需要颜色条
    plt.imshow(img_arr_diff, cmap='bwr')
    plt.clim(-0.5, 0.5)  # 设置颜色范围

    # 完全去除坐标轴
    plt.axis('off')

    # 去除图片四周的黑色边框
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # 保存图片，去除空白区域
    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.clf()


def kl_loss(dist, latent_var):
    mu = dist
    var = latent_var
    logvar = torch.log(var)
    loss = 0.5*torch.mean(var+mu.pow(2)-1-logvar)
    return loss
