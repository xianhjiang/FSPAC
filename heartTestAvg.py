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

if __name__ == "__main__":
    idx = 16
    setup_seed(cfg.SEED)
    utils.remkdir(cfg.TEST_PATH)

    stn = SpatialTransformer(cfg.HEIGHT).to(device)
    seg_stn = SpatialTransformer(cfg.HEIGHT, mode='nearest').to(device)


    Dataset = eval('data_util.{}.Dataset'.format(cfg.IMAGE_TYPE))
    dataset = Dataset(split_path='datasets/%s.json' % cfg.IMAGE_TYPE, paired=False, mode=cfg.DATA_TYPE)
    test_loader = dataset.generator(2, batch_size=1, loop=False, aug=False)

    brain = SPAC(stn, seg_stn, device)
    brain.load_model('actor', cfg.ACTOR_MODEL)
    brain.load_model('planner', cfg.PLANNER_MODEL)
    brain.load_model('critic1', cfg.CRITIC1_MODEL)
    brain.load_model('critic2', cfg.CRITIC2_MODEL)

    brain.eval(brain.actor)
    brain.eval(brain.critic1)
    brain.eval(brain.critic2)
    brain.eval(brain.planner)

    avg_dices = []
    dices = []
    seg1_dices = []
    seg2_dices = []
    seg3_dices = []
    times = []
    latents = []
    labs = []
    jacobian_dets = []

    for i, item in enumerate(test_loader):
        fixed = item['fixed']
        fixed_seg = item['fixed_seg']
        moving = item['moving']
        moving_seg = item['moving_seg']

        if i % 1 == 0:
            # cv2.imwrite('{}/{}-fixed.bmp'.format(cfg.TEST_PATH, i), utils.numpy_im_round(fixed)[:, :, idx])
            # cv2.imwrite('{}/{}-moving.bmp'.format(cfg.TEST_PATH, i), utils.numpy_im_round(moving)[:, :, idx])
            vutils.save_image(fixed[:, :, :, :, idx].data, cfg.TEST_PATH + '/{}-fixed.bmp'.format(i), normalize=False)
            vutils.save_image(moving[:, :, :, :, idx].data, cfg.TEST_PATH + '/{}-moving.bmp'.format(i), normalize=False)


        fixed_seg = utils.numpy_im_round(fixed_seg, 1)

        # init_score = utils.dice(fixed_seg > 0, utils.numpy_im_round(moving_seg) > 0, [1])
        init_scores = utils.dice(fixed_seg, utils.numpy_im_round(moving_seg), [1, 2, 3])
        init_score = np.mean(init_scores)
        # moving_seg_tensor = torch.from_numpy(moving_seg)
        # moving_seg = moving_seg_tensor.to(device)[None, ...]
        moving_seg = moving_seg.to(device)
        # fixed = fixed.to(device).unsqueeze(0)
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
            vutils.save_image(moved[:, :, :, :, idx].data, 'resultProcess/{}-moved-{}.bmp'.format(i, step + 1),
                              normalize=True)

            warped_seg_step = utils.numpy_im_round(seg_stn(moving_seg, pred.to(moving_seg.dtype)), 1, device)
            scores = utils.dice(fixed_seg, warped_seg_step, [1, 2, 3])

            score = np.mean(scores)
            if score >= init_score:
                opt_pred = pred
                init_score = score

            step += 1

        toc = time.time()
        # warped_im = utils.numpy_im_round(stn(moving, pred), device=device)
        if opt_pred is not None:
            pred = opt_pred
        warped_im = stn(moving, pred)
        warped_seg = utils.numpy_im_round(seg_stn(moving_seg, pred.to(moving_seg.dtype)), 1, device)
        labels = np.unique(fixed_seg)[1:]

        # 这是算重叠dice
        scores = utils.dice(fixed_seg > 0, warped_seg > 0, [1])
        score = np.mean(scores)
        dices.append(score)

        # 这里算分割dice
        seg_scores = utils.dice(fixed_seg, warped_seg, [1, 2, 3])

        # for idx, score in enumerate(seg_scores):
        #     print("seg{}数据的dice是：{:.15f}".format(idx+1, score.item()))
        seg1 = seg_scores[0]
        seg2 = seg_scores[1]
        seg3 = seg_scores[2]
        seg1_dices.append(seg1)
        seg2_dices.append(seg2)
        seg3_dices.append(seg3)

        avg_score = (seg1 + seg2 + seg3) / 3
        avg_dices.append(avg_score)
        times.append(toc - tic)
        # 保存形变场
        flowPath = "./result/" + str(i) + ".png"
        utils.plot_flow_grid(pred, interval=2, path=flowPath)

        flow = utils.numpy(pred.squeeze(), device=device)
        jacobian_det = utils.jacobian_determinant(np.transpose(flow, (1, 2, 3, 0)))
        det = np.sum(jacobian_det <= 0) / np.prod(fixed.shape)
        print('det < 0: {:.10f}'.format(det))
        jacobian_dets.append(det)
        # jacobian_dets.append(jacobian_det)

        if i % 1 == 0:
            # cv2.imwrite('{}/{}-pred_img.bmp'.format(cfg.TEST_PATH, i), warped_im[:, :, idx])
            # warped_im = torch.tensor(warped_im).float()
            differencePath = "./result/" + str(i) + "-difference" + ".png"
            fixed_np = fixed.cpu().numpy().squeeze()
            warped_im_np = warped_im.cpu().numpy().squeeze()


            # utils.plot_difference(fixed_seg, warped_seg, differencePath)
            utils.plot_difference(fixed_np, warped_im_np, differencePath)

            warped_im = torch.tensor(warped_im).clone().detach().float()
            vutils.save_image(warped_im[:, :, :, :, idx].data, cfg.TEST_PATH + '/{}-pred.bmp'.format(i), normalize=True)

            vis_seg = utils.render_image_with_mask(utils.numpy_im_round(fixed, device=device)[:, :, idx],
                                                   warped_seg[:, :, idx], color=1, colorful=True)
            # cv2.imwrite('{}/{}-vis_pre.png'.format(cfg.TEST_PATH, i), vis_seg)
            utils.plot_label(fixed_np * 255., warped_seg, colorful=True, path='{}/{}-vis_pre.png'.format(cfg.TEST_PATH, i))


            vis_seg = utils.render_image_with_mask(utils.numpy_im_round(fixed, device=device)[:, :, idx],
                                                   fixed_seg[:, :, idx], color=0, colorful=True)
            # cv2.imwrite('{}/{}-vis_gt.png'.format(cfg.TEST_PATH, i), vis_seg)

            utils.plot_label(fixed_np * 255., fixed_seg, colorful=True, path='{}/{}-vis_gt.png'.format(cfg.TEST_PATH, i))


        ndices = np.array(dices)
        print('data-{}: avg time: {:.4f}, dice: {:.15f}, seg1: {:.4f}, seg2: {:.4f}, seg3: {:.4f}, det: {:.4f} final dice: {:.4f}({:.4f})'
              .format(i, np.mean(times), score, seg1, seg2, seg3, det, np.mean(ndices), np.std(ndices)))
        # if i == 49:
        #     break

    print('avg time: {:.4f}, final dice: {:.4f}({:.4f}), avg seg1: {:.4f}, avg seg2: {:.4f}, avg seg3: {:.4f}, all avg: {:.4f}, 50%: {:.4f}, 90%: {:.4f}'
          .format(np.mean(times), np.mean(ndices), np.std(ndices), np.mean(seg1_dices), np.mean(seg2_dices), np.mean(seg3_dices), (np.mean(seg1_dices) + np.mean(seg2_dices) + np.mean(seg3_dices)) / 3,
                  np.median(ndices), np.percentile(ndices, 90)))

    print('final jacobian determinant: {:.15f}({:.15f})'.format(np.mean(jacobian_dets), np.std(jacobian_dets)))
    print('avg_dices: {:.4f}({:.4f})'.format(np.mean(avg_dices), np.std(avg_dices)))
