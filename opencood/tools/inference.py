import argparse
import os
import time

import torch
import open3d as o3d
import cv2
import numpy as np
from torch.utils.data import DataLoader

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils, infrence_utils
from opencood.data_utils.datasets import build_dataset
from opencood.visualization import vis_utils
from opencood.utils import eval_utils, box_utils


def test_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Continued training path')
    parser.add_argument('--fusion_method', required=True, type=str,
                        default='late',
                        help='nofusion, late, early or intermediate')
    parser.add_argument('--show_vis', action='store_true',
                        help='whether to show image visualization result')
    parser.add_argument('--show_sequence', action='store_true',
                        help='whether to show video visualization result.'
                             'it can note be set true with show_vis together ')
    parser.add_argument('--save_vis', action='store_true',
                        help='whether to save visualization result')
    parser.add_argument('--save_npy', action='store_true',
                        help='whether to save prediction and gt result')
    parser.add_argument('--save_evibev', action='store_true',
                        help='set true to skip evaluation for now and save result in evibev format for later evaluation')
    parser.add_argument('--isSim', action='store_true',
                        help='whether to save prediction and gt result'
                             'in npy file')
    opt = parser.parse_args()
    return opt


def main():
    opt = test_parser()
    assert opt.fusion_method in ['late', 'early', 'intermediate', 'nofusion']
    assert not (opt.show_vis and opt.show_sequence), \
        'you can only visualize ' \
        'the results in single ' \
        'image mode or video mode'

    hypes = yaml_utils.load_yaml(None, opt)

    print('Dataset Building')
    opencood_dataset = build_dataset(hypes, visualize=True, train=False,
                                     isSim=opt.isSim)
    data_loader = DataLoader(opencood_dataset,
                             batch_size=1,
                             num_workers=16,
                             collate_fn=opencood_dataset.collate_batch_test,
                             shuffle=False,
                             pin_memory=False,
                             drop_last=False)

    print('Creating Model')
    model = train_utils.create_model(hypes)
    # we assume gpu is necessary
    if torch.cuda.is_available():
        model.cuda()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Loading Model from checkpoint')
    saved_path = opt.model_dir
    _, model = train_utils.load_saved_model(saved_path, model)
    model.eval()

    # Create the dictionary for evaluation
    result_stat = {0.5: {'tp': [], 'fp': [], 'gt': 0},
                   0.7: {'tp': [], 'fp': [], 'gt': 0}}
    result_stat_short = {0.5: {'tp': [], 'fp': [], 'gt': 0},
                         0.7: {'tp': [], 'fp': [], 'gt': 0}}
    result_stat_middle = {0.5: {'tp': [], 'fp': [], 'gt': 0},
                          0.7: {'tp': [], 'fp': [], 'gt': 0}}
    result_stat_long = {0.5: {'tp': [], 'fp': [], 'gt': 0},
                        0.7: {'tp': [], 'fp': [], 'gt': 0}}

    if opt.show_sequence:
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        vis.get_render_option().background_color = [0.05, 0.05, 0.05]
        vis.get_render_option().point_size = 1.0
        vis.get_render_option().line_width = 10
        vis.get_render_option().show_coordinate_frame = True

        # used to visualize lidar points
        vis_pcd = o3d.geometry.PointCloud()
        # used to visualize object bounding box, maximum 50
        vis_aabbs_gt = []
        vis_aabbs_pred = []
        for _ in range(500):
            vis_aabbs_gt.append(o3d.geometry.TriangleMesh())
            vis_aabbs_pred.append(o3d.geometry.TriangleMesh())

    for i, batch_data in enumerate(data_loader):
        print(i)
        with torch.no_grad():
            torch.cuda.synchronize()
            batch_data = train_utils.to_device(batch_data, device)
            if opt.fusion_method == 'nofusion':
                res = infrence_utils.inference_no_fusion(batch_data,
                                                       model,
                                                       opencood_dataset)
            else:
                fusion_fn = getattr(infrence_utils,
                                    f'inference_{opt.fusion_method}_fusion',
                                    None)
                if fusion_fn is None:
                    raise NotImplementedError('Only early, late and intermediate'
                                              'fusion is supported.')
                else:
                    res = fusion_fn(batch_data, model, opencood_dataset)

            if len(res) == 3:
                pred_box_tensor, pred_score, gt_box_tensor = res
                pred_bev = None
            elif len(res) == 4:
                pred_box_tensor, pred_score, pred_bev, gt_box_tensor = res

            if opt.save_evibev:
                pred_boxes = box_utils.corner_to_center(pred_box_tensor.detach().cpu().numpy())
                gt_boxes = box_utils.corner_to_center(gt_box_tensor.detach().cpu().numpy())
                evidence = torch.stack([bev.permute(1, 2, 0) for bev in pred_bev], dim=0)
                device = pred_box_tensor.device
                cur_dict = {
                    'frame_id': batch_data['ego']['frame_id'],
                    'detection': {
                        'pred_boxes': torch.from_numpy(pred_boxes).to(device),
                        'pred_scores': pred_score,
                    },
                    'target_boxes': torch.from_numpy(gt_boxes).to(device),
                    'distr_object': {
                        'evidence': evidence,
                        'obs_mask': None,
                        'Nall': None,
                        'Nsel': None
                    }
                }
                frame_id = '_'.join(batch_data['ego']['frame_id'][0])
                torch.save(cur_dict, os.path.join(opt.model_dir, f'{frame_id}.pth'))
            else:
                # overall calculating
                eval_utils.caluclate_tp_fp(pred_box_tensor,
                                           pred_score,
                                           gt_box_tensor,
                                           result_stat,
                                           0.5)
                eval_utils.caluclate_tp_fp(pred_box_tensor,
                                           pred_score,
                                           gt_box_tensor,
                                           result_stat,
                                           0.7)
                # short range
                eval_utils.caluclate_tp_fp(pred_box_tensor,
                                           pred_score,
                                           gt_box_tensor,
                                           result_stat_short,
                                           0.5,
                                           left_range=0,
                                           right_range=30)
                eval_utils.caluclate_tp_fp(pred_box_tensor,
                                           pred_score,
                                           gt_box_tensor,
                                           result_stat_short,
                                           0.7,
                                           left_range=0,
                                           right_range=30)

                # middle range
                eval_utils.caluclate_tp_fp(pred_box_tensor,
                                           pred_score,
                                           gt_box_tensor,
                                           result_stat_middle,
                                           0.5,
                                           left_range=30,
                                           right_range=50)
                eval_utils.caluclate_tp_fp(pred_box_tensor,
                                           pred_score,
                                           gt_box_tensor,
                                           result_stat_middle,
                                           0.7,
                                           left_range=30,
                                           right_range=50)

                # right range
                eval_utils.caluclate_tp_fp(pred_box_tensor,
                                           pred_score,
                                           gt_box_tensor,
                                           result_stat_long,
                                           0.5,
                                           left_range=50,
                                           right_range=100)
                eval_utils.caluclate_tp_fp(pred_box_tensor,
                                           pred_score,
                                           gt_box_tensor,
                                           result_stat_long,
                                           0.7,
                                           left_range=50,
                                           right_range=100)

            if opt.save_npy:
                npy_save_path = os.path.join(opt.model_dir, 'npy')
                if not os.path.exists(npy_save_path):
                    os.makedirs(npy_save_path)
                infrence_utils.save_prediction_gt(pred_box_tensor,
                                                  gt_box_tensor,
                                                  batch_data['ego'][
                                                      'origin_lidar'][0],
                                                  i,
                                                  npy_save_path)

            if opt.show_vis or opt.save_vis:
                vis_save_path = ''
                if opt.save_vis:
                    vis_save_path = os.path.join(opt.model_dir, 'vis')
                    if not os.path.exists(vis_save_path):
                        os.makedirs(vis_save_path)
                    vis_save_path = os.path.join(vis_save_path, '%05d.png' % i)

                opencood_dataset.visualize_result(pred_box_tensor,
                                                  gt_box_tensor,
                                                  batch_data['ego'][
                                                      'origin_lidar'][0],
                                                  opt.show_vis,
                                                  vis_save_path,
                                                  dataset=opencood_dataset,
                                                  bev_map=pred_bev)

            if opt.show_sequence:
                if pred_bev is not None:
                    bev_map_np = pred_bev[0].permute(1, 2, 0).detach().cpu().numpy()
                    img = np.zeros((*bev_map_np.shape[:2], 3)).astype(np.int8)
                    img[..., 2] = np.clip(np.round(bev_map_np[..., 0] * 255), a_min=0, a_max=155).astype(np.int8)
                    cv2.imshow('bev_map', img[::-1, ::-1])
                    cv2.waitKey(1)
                pcd, pred_o3d_box, gt_o3d_box = \
                    vis_utils.visualize_inference_sample_dataloader(
                        pred_box_tensor,
                        gt_box_tensor,
                        batch_data['ego']['origin_lidar'][0],
                        vis_pcd,
                        mode='constant'
                    )
                if i == 0:
                    vis.add_geometry(pcd)
                    vis_utils.linset_assign_list(vis,
                                                 vis_aabbs_pred,
                                                 pred_o3d_box,
                                                 update_mode='add')

                    vis_utils.linset_assign_list(vis,
                                                 vis_aabbs_gt,
                                                 gt_o3d_box,
                                                 update_mode='add')

                vis_utils.linset_assign_list(vis,
                                             vis_aabbs_pred,
                                             pred_o3d_box)
                vis_utils.linset_assign_list(vis,
                                             vis_aabbs_gt,
                                             gt_o3d_box)
                vis.update_geometry(pcd)
                vis.poll_events()
                vis.update_renderer()
                time.sleep(0.001)

    if not opt.save_evibev:
        eval_utils.eval_final_results(result_stat,
                                      opt.model_dir)
        eval_utils.eval_final_results(result_stat_short,
                                      opt.model_dir,
                                      "short")
        eval_utils.eval_final_results(result_stat_middle,
                                      opt.model_dir,
                                      "middle")
        eval_utils.eval_final_results(result_stat_long,
                                      opt.model_dir,
                                      "long")
    if opt.show_sequence:
        vis.destroy_window()


if __name__ == '__main__':
    main()
