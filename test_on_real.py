import torch
import os
import sys
import yaml
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

# 添加本地模块路径
sys.path.append(os.getcwd())

from model.super_retina_multimodal import SuperRetinaMultimodal
from common.common_util import nms, sample_keypoint_desc
from dataset.CFFA.cffa_real_dataset import CFRegistrationDataset as CFFARealDataset
from dataset.CF_OCT.cfoct_real_dataset import CFRegistrationDataset as CFOCTRealDataset

def draw_matches(img1, kps1, img2, kps2, matches, save_path):
    """在两张图像之间绘制匹配连线"""
    if torch.is_tensor(img1): img1 = (img1.cpu().numpy() * 255).astype(np.uint8)
    if torch.is_tensor(img2): img2 = (img2.cpu().numpy() * 255).astype(np.uint8)
    if img1.ndim == 3: img1 = img1.squeeze()
    if img2.ndim == 3: img2 = img2.squeeze()
    kp1_cv = [cv2.KeyPoint(x=float(pt[0]), y=float(pt[1]), size=10) for pt in kps1]
    kp2_cv = [cv2.KeyPoint(x=float(pt[0]), y=float(pt[1]), size=10) for pt in kps2]
    out_img = cv2.drawMatches(img1, kp1_cv, img2, kp2_cv, matches, None, 
                             flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imwrite(save_path, out_img)

def compute_checkerboard(img1, img2, n_grid=3):
    h, w = img1.shape[:2]
    grid_h, grid_w = h // n_grid, w // n_grid
    checkerboard = np.zeros_like(img1)
    for i in range(n_grid):
        for j in range(n_grid):
            y_s, y_e = i * grid_h, (i + 1) * grid_h
            x_s, x_e = j * grid_w, (j + 1) * grid_w
            checkerboard[y_s:y_e, x_s:x_e] = img1[y_s:y_e, x_s:x_e] if (i + j) % 2 == 0 else img2[y_s:y_e, x_s:x_e]
    return checkerboard

def cal_metrics_ref(pts_f, pts_m, H):
    if pts_f is None or pts_m is None or H is None: return None, None
    pts_m_warped = cv2.perspectiveTransform(pts_m.reshape(-1, 1, 2).astype(np.float32), H).reshape(-1, 2)
    diff = pts_f - pts_m_warped
    distances = np.sqrt(np.sum(diff**2, axis=1))
    return distances.mean(), np.mean(diff**2)

def compute_auc_ref(avg_dists, limit=25):
    if not avg_dists: return 0.0
    avg_dists = np.array(avg_dists)
    accum_s = 0
    for i in range(1, limit + 1):
        accum_s += np.sum(avg_dists < i) * 100 / len(avg_dists)
    return accum_s / (limit * 100)

def compute_repeatability(kps1, kps2, H_gt, threshold=5, img_shape=(768, 768)):
    """
    计算检测器重复性 (Rep) - 遵循 Oxford [5] 标准：
    只在两图共有的区域（Shared Region）内计算关键点重合率
    """
    if len(kps1) == 0 or len(kps2) == 0 or H_gt is None: return 0.0
    
    kps1, kps2, H_gt = kps1.astype(np.float32), kps2.astype(np.float32), H_gt.astype(np.float32)
    H_inv = np.linalg.inv(H_gt)
    h, w = img_shape

    # 1. 将 A 点投影到 B，剔除超出 B 范围的点
    kps1_warped = cv2.perspectiveTransform(kps1.reshape(-1, 1, 2), H_gt).reshape(-1, 2)
    mask1 = (kps1_warped[:, 0] >= 0) & (kps1_warped[:, 0] < w) & (kps1_warped[:, 1] >= 0) & (kps1_warped[:, 1] < h)
    kps1_shared = kps1_warped[mask1]

    # 2. 将 B 点投影回 A，剔除超出 A 范围的点
    kps2_warped_back = cv2.perspectiveTransform(kps2.reshape(-1, 1, 2), H_inv).reshape(-1, 2)
    mask2 = (kps2_warped_back[:, 0] >= 0) & (kps2_warped_back[:, 0] < w) & (kps2_warped_back[:, 1] >= 0) & (kps2_warped_back[:, 1] < h)
    kps2_shared = kps2[mask2]

    n1, n2 = len(kps1_shared), len(kps2_shared)
    if n1 == 0 or n2 == 0: return 0.0

    # 3. 计算最近邻匹配
    dists = np.linalg.norm(kps1_shared[:, None, :] - kps2_shared[None, :, :], axis=2)
    n_corr = np.sum(np.min(dists, axis=1) < threshold) if dists.size > 0 else 0
    
    # 按照论文标准返回百分比
    return (n_corr / (n1 + n2 - n_corr)) * 100

def test_on_real():
    parser = argparse.ArgumentParser(description="SuperRetina 真实数据测试脚本")
    parser.add_argument('-n', '--name', type=str, required=True, help='实验名称')
    parser.add_argument('-latestcheckpoint', action='store_true', help='使用最新的 checkpoint')
    parser.add_argument('--mode', type=str, choices=['cffa', 'cfoct'], default='cffa', help='模式')
    args = parser.parse_args()

    exp_name, mode = args.name, args.mode
    checkpoint_type = 'latestpoint' if args.latestcheckpoint else 'bestcheckpoint'
    checkpoint_path = f'./save/{mode}/{exp_name}/{checkpoint_type}/checkpoint.pth'
    output_root = f'./save/test_on_real/{mode}/{exp_name}'
    os.makedirs(output_root, exist_ok=True)

    config_path = './config/train_multimodal.yaml'
    with open(config_path) as f: config = yaml.safe_load(f)
    model_config = {**config['MODEL'], **config['PKE'], **config['DATASET'], **config['VALUE_MAP']}

    if mode == 'cffa':
        test_set = CFFARealDataset(root_dir='/data/student/Fengjunming/SuperRetina/dataset/CFFA', registration_type='cffa')
    else:
        test_set = CFOCTRealDataset(root_dir='/data/student/Fengjunming/SuperRetina/dataset/CF_OCT', registration_type='cfoct')
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SuperRetinaMultimodal(model_config, device=device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['net'])
    model.eval()

    results_log, auc_record, mse_record, rep_record, mir_record = [], [], [], [], []
    success_3, success_5 = 0, 0

    with torch.no_grad():
        for fixed_imgs, moving_imgs, _, _, sample_ids, gt_pts in tqdm(test_loader, desc="Testing"):
            fixed_imgs, moving_imgs = fixed_imgs.to(device), moving_imgs.to(device)
            sample_id = sample_ids[0]
            sample_save_dir = os.path.join(output_root, sample_id)
            os.makedirs(sample_save_dir, exist_ok=True)

            det_f, desc_f = model.network(fixed_imgs)
            det_m, desc_m = model.network(moving_imgs)
            kps_f = nms(det_f, nms_thresh=0.01, nms_size=5)[0]
            kps_m = nms(det_m, nms_thresh=0.01, nms_size=5)[0]
            
            good = []
            if len(kps_f) >= 4 and len(kps_m) >= 4:
                d_f = sample_keypoint_desc(kps_f[None], desc_f, s=8)[0].permute(1, 0).cpu().numpy()
                d_m = sample_keypoint_desc(kps_m[None], desc_m, s=8)[0].permute(1, 0).cpu().numpy()
                matches = cv2.BFMatcher().knnMatch(d_f, d_m, k=2)
                good = [m for m, n in matches if m.distance < 0.8 * n.distance]

            fix_np = (fixed_imgs[0, 0].cpu().numpy() * 255).astype(np.uint8)
            mov_np = (moving_imgs[0, 0].cpu().numpy() * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(sample_save_dir, 'fixed.png'), fix_np)
            
            M, pair_rmse, pair_avg_dist = None, None, None
            if len(good) >= 4:
                src_p = np.float32([kps_f[m.queryIdx].cpu().numpy() for m in good]).reshape(-1, 1, 2)
                dst_p = np.float32([kps_m[m.trainIdx].cpu().numpy() for m in good]).reshape(-1, 1, 2)
                M, mask = cv2.findHomography(dst_p, src_p, cv2.RANSAC, 5.0)
                
                if M is not None:
                    mir_record.append((np.sum(mask) / len(good)) * 100)
                    reg_img = cv2.warpPerspective(mov_np, M, (fix_np.shape[1], fix_np.shape[0]))
                    cv2.imwrite(os.path.join(sample_save_dir, 'registered.png'), reg_img)
                    cv2.imwrite(os.path.join(sample_save_dir, 'checkerboard.png'), compute_checkerboard(fix_np, reg_img))
                    
                    pts_f_gt, pts_m_gt = gt_pts['fixed'][0].numpy() if gt_pts['fixed'] is not None else None, gt_pts['moving'][0].numpy() if gt_pts['moving'] is not None else None
                    if pts_f_gt is not None:
                        avg_dist, mse = cal_metrics_ref(pts_f_gt, pts_m_gt, M)
                        if avg_dist is not None:
                            auc_record.append(avg_dist); mse_record.append(mse)
                            pair_avg_dist, pair_rmse = avg_dist, np.sqrt(mse)
                            if pair_avg_dist < 3: success_3 += 1
                            if pair_avg_dist < 5: success_5 += 1
                        H_gt, _ = cv2.findHomography(pts_f_gt, pts_m_gt, cv2.LMEDS)
                        rep_record.append(compute_repeatability(kps_f.cpu().numpy(), kps_m.cpu().numpy(), H_gt))

            log_str = f"ID: {sample_id} | RMSE: {f'{pair_rmse:.2f}' if pair_rmse is not None else 'FAIL'} | AvgDist: {f'{pair_avg_dist:.2f}' if pair_avg_dist is not None else 'FAIL'}"
            results_log.append(log_str)
            draw_matches(fix_np, kps_f.cpu().numpy(), mov_np, kps_m.cpu().numpy(), good, os.path.join(sample_save_dir, 'matches.png'))

    total_pairs = len(test_loader)
    summary_path = os.path.join(output_root, 'metrics_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("\n".join(results_log) + "\n" + "-" * 40 + "\n")
        f.write(f"SR_ME (e=3): {(success_3 / total_pairs) * 100:.2f}%\n")
        f.write(f"SR_MAE (e=5): {(success_5 / total_pairs) * 100:.2f}%\n")
        f.write(f"Rep (e=5): {np.mean(rep_record) if rep_record else 0:.2f}%\n")
        f.write(f"MIR (e=5): {np.mean(mir_record) if mir_record else 0:.2f}%\n")
        f.write(f"Overall RMSE: {np.sqrt(np.mean(mse_record)) if mse_record else 0:.2f}\n")
        f.write(f"Overall mAUC: {compute_auc_ref(auc_record):.4f}\n")
    
    print(f"\nFinal Metrics -> SR_ME(3): {(success_3/total_pairs)*100:.2f}% | SR_MAE(5): {(success_5/total_pairs)*100:.2f}% | Rep: {np.mean(rep_record):.2f}% | MIR: {np.mean(mir_record):.2f}%")
