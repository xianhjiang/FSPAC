import torch
import random
import numpy as np
import copy
import cv2
import time
import torchvision.utils as vutils

import utils
import data_util.heart

from config import Config as cfg
from brain import SPAC
from networks import SpatialTransformer

if torch.cuda.is_available():
    device = torch.device(f'cuda:{cfg.GPU_ID}')
    torch.cuda.set_device(cfg.GPU_ID)
else:
    device = torch.device('cpu')

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def run_registration_test(dataset, brain, device):
    idx = 8
    setup_seed(cfg.SEED)

    if not os.path.exists(cfg.TEST_PATH):
        utils.remkdir(cfg.TEST_PATH)

    stn = SpatialTransformer(cfg.HEIGHT).to(device)
    seg_stn = SpatialTransformer(cfg.HEIGHT, mode='nearest').to(device)

    test_loader = dataset.generator(2, batch_size=1, loop=False, aug=False)

    brain.eval(brain.actor)
    brain.eval(brain.critic1)
    brain.eval(brain.critic2)
    brain.eval(brain.planner)

    dices = []
    seg1_dices = []
    seg2_dices = []
    seg3_dices = []
    avg_seg_dices = []
    times = []
    latents = []
    labs = []
    jacobian_dets = []

    for i, item in enumerate(test_loader):
        fixed = item['fixed']
        fixed_seg = item['fixed_seg']
        moving = item['moving']
        moving_seg = item['moving_seg']

        # if i % 1 == 0:
        #     vutils.save_image(fixed[:, :, :, :, idx].data, cfg.TEST_PATH + '/{}-fixed.bmp'.format(i), normalize=True)
        #     vutils.save_image(moving[:, :, :, :, idx].data, cfg.TEST_PATH + '/{}-moving.bmp'.format(i), normalize=True)

        fixed_seg = utils.numpy_im(fixed_seg, 1)

        init_scores = utils.dice(fixed_seg, utils.numpy_im_round(moving_seg), [1, 2, 3])
        init_score = np.mean(init_scores)
        moving_seg = moving_seg.to(device)
        fixed = fixed.to(device)
        moving = moving.to(device)

        moved = copy.deepcopy(moving)
        pred = None
        opt_pred = None
        step = 0
        tic = time.time()

        while step < 10:
            state = torch.cat([fixed, moved], dim=1)
            latent, flow = brain.choose_action(state, test=True)
            latents.append(latent.cpu().numpy())
            labs.append(i)
            pred = flow if pred is None else stn(pred, flow) + flow
            moved = stn(moving, pred)
            # vutils.save_image(moved[:, :, :, :, idx].data, 'resultProcess/{}-moved-{}.bmp'.format(i, step + 1),
            #                   normalize=True)

            warped_seg_step = utils.numpy_im(seg_stn(moving_seg, pred.to(moving_seg.dtype)), 1, device)
            scores = utils.dice(fixed_seg, warped_seg_step, [1, 2, 3])
            score = np.mean(scores)
            if score >= init_score:
                opt_pred = pred
                init_score = score

            step += 1

        toc = time.time()
        if opt_pred is not None:
            pred = opt_pred
        warped_im = stn(moving, pred)
        warped_seg = utils.numpy_im(seg_stn(moving_seg, pred.to(moving_seg.dtype)), 1, device)
        labels = np.unique(fixed_seg)[1:]

        scores = utils.dice(fixed_seg > 0, warped_seg > 0, [1])
        score = np.mean(scores)
        dices.append(score)

        seg_scores = utils.dice(fixed_seg, warped_seg, [1, 2, 3])
        seg1_dices.append(seg_scores[0])
        seg2_dices.append(seg_scores[1])
        seg3_dices.append(seg_scores[2])

        avg_dice = np.mean(seg_scores)

        avg_seg_dices.append(avg_dice)

        times.append(toc - tic)

        flow = utils.numpy(pred.squeeze(), device=device)
        jacobian_det = np.mean(utils.jacobian_determinant(np.transpose(flow, (1, 2, 3, 0))))
        jacobian_dets.append(jacobian_det)

        # if i % 1 == 0:
        #     warped_im = torch.tensor(warped_im).clone().detach().float()
        #     vutils.save_image(warped_im[:, :, :, :, idx].data, cfg.TEST_PATH + '/{}-pred_img.bmp'.format(i), normalize=True)
        #
        #     vis_seg = utils.render_image_with_mask(utils.numpy_im(fixed, device=device)[:, :, idx],
        #                                            warped_seg[:, :, idx], color=1)
        #     cv2.imwrite('{}/{}-vis_pre.png'.format(cfg.TEST_PATH, i), vis_seg)
        #
        #     vis_seg = utils.render_image_with_mask(utils.numpy_im(fixed, device=device)[:, :, idx],
        #                                            fixed_seg[:, :, idx], color=0)
        #     cv2.imwrite('{}/{}-vis_gt.png'.format(cfg.TEST_PATH, i), vis_seg)

        # if i == 49:
        #     break

    ndices = np.array(dices)

    brain.train(brain.actor)
    brain.train(brain.critic1)
    brain.train(brain.critic2)
    brain.train(brain.planner)

    return {
        'avg_time': np.mean(times),
        'final_dice': np.mean(ndices),
        'final_dice_std': np.std(ndices),
        'seg1_dice': np.mean(seg1_dices),
        'seg2_dice': np.mean(seg2_dices),
        'seg3_dice': np.mean(seg3_dices),
        'avg_seg_dice':np.mean(avg_seg_dices),
        'median_dice': np.median(ndices),
        '90th_percentile_dice': np.percentile(ndices, 90),
        'jacobian_determinant_mean': np.mean(jacobian_dets),
        'jacobian_determinant_std': np.std(jacobian_dets)
    }

if __name__ == "__main__":
    Dataset = eval('data_util.{}.Dataset'.format(cfg.IMAGE_TYPE))
    dataset = Dataset(split_path='datasets/%s.json' % cfg.IMAGE_TYPE, paired=False, mode=cfg.DATA_TYPE)
    test_loader = dataset.generator(2, batch_size=1, loop=True, aug=False)

    results = run_registration_test(test_loader, device)
    print("Results:")
    print(f"Average time: {results['avg_time']:.4f}")
    print(f"Final dice score: {results['final_dice']:.4f} (std: {results['final_dice_std']:.4f})")
    print(f"Seg1 dice score: {results['seg1_dice']:.4f}")
    print(f"Seg2 dice score: {results['seg2_dice']:.4f}")
    print(f"Seg3 dice score: {results['seg3_dice']:.4f}")
    print(f"Median dice score: {results['median_dice']:.4f}")
    print(f"90th percentile dice score: {results['90th_percentile_dice']:.4f}")
    print(f"Jacobian determinant mean: {results['jacobian_determinant_mean']:.4f} (std: {results['jacobian_determinant_std']:.4f})")
