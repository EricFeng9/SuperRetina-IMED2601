"""
基于v6版本: Dual-Path Hybrid (End-to-End Multimodal)
1. 架构: Dual-Path Encoder (Fix/Mov 独立权重), 解决模态差异
2. 输入: 原始图像 (Raw Input), 移除反色 (No Inversion)
3. 训练策略:
   - GT Anchor Alignment: 强制对齐 GT 分叉点特征
   - Self-Supervised PKE with GT Init: GT点作为初始化种子
   - Mask-Constrained Detector: 抑制背景误检
"""

import torch
import os
import sys
import yaml
import shutil
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import argparse
import random

# 添加本地模块路径
sys.path.append(os.getcwd())

# 使用新数据集脚本
from dataset.FIVES_extract_v2.FIVES_extract_v2 import MultiModalDataset
from dataset.CF_OCTA_v2_repaired.cf_octa_v2_repaired_dataset import CFOCTADataset
from dataset.operation_pre_filtered_cffa.operation_pre_filtered_cffa_dataset import CFFADataset
from dataset.operation_pre_filtered_cfoct.operation_pre_filtered_cfoct_dataset import CFOCTDataset
from dataset.operation_pre_filtered_octfa.operation_pre_filtered_octfa_dataset import OCTFADataset
from model.super_retina_multimodal import SuperRetinaMultimodal
from common.train_util import value_map_load, value_map_save, affine_images
from common.common_util import nms, sample_keypoint_desc
from common.vessel_keypoint_extractor import extract_vessel_keypoints, extract_vessel_keypoints_fallback
from torchvision import transforms
from gen_data_enhance_v2 import apply_domain_randomization, save_batch_visualization


# ============================================================================
# 域随机化增强 (Domain Randomization) - 使用 gen_data_enhance_v2.py
# ============================================================================

def compute_corner_error(H_est, H_gt, height, width):
    """计算角点平均误差 (MACE)"""

    corners = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
    corners_homo = np.concatenate([corners, np.ones((4, 1), dtype=np.float32)], axis=1)
    
    # GT 变换后的角点
    corners_gt_homo = (H_gt @ corners_homo.T).T
    corners_gt = corners_gt_homo[:, :2] / (corners_gt_homo[:, 2:] + 1e-6)
    
    # 预测变换后的角点
    corners_est_homo = (H_est @ corners_homo.T).T
    corners_est = corners_est_homo[:, :2] / (corners_est_homo[:, 2:] + 1e-6)
    
    try:
        errors = np.sqrt(np.sum((corners_est - corners_gt)**2, axis=1))
        mace = np.mean(errors)
    except:
        mace = float('inf')
    return mace




def compute_checkerboard(img1, img2, n_grid=4):
    """计算棋盘格可视化"""
    if img1.ndim == 3: img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY) if img1.shape[2] == 3 else img1.squeeze()
    if img2.ndim == 3: img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY) if img2.shape[2] == 3 else img2.squeeze()
    h, w = img1.shape[:2]
    if img2.shape[:2] != (h, w): img2 = cv2.resize(img2, (w, h))
    grid_h, grid_w = h // n_grid, w // n_grid
    checkerboard = np.zeros_like(img1)
    for i in range(n_grid):
        for j in range(n_grid):
            y_s, y_e = i * grid_h, (i + 1) * grid_h if i < n_grid - 1 else h
            x_s, x_e = j * grid_w, (j + 1) * grid_w if j < n_grid - 1 else w
            checkerboard[y_s:y_e, x_s:x_e] = img1[y_s:y_e, x_s:x_e] if (i + j) % 2 == 0 else img2[y_s:y_e, x_s:x_e]
    return checkerboard

def draw_matches(img1, kps1, img2, kps2, matches, save_path):
    """
    在两张图像之间绘制匹配连线
    """
    if torch.is_tensor(img1): img1 = (img1.cpu().numpy() * 255).astype(np.uint8)
    if torch.is_tensor(img2): img2 = (img2.cpu().numpy() * 255).astype(np.uint8)
    
    if img1.ndim == 3: img1 = img1.squeeze()
    if img2.ndim == 3: img2 = img2.squeeze()
    
    kp1_cv = [cv2.KeyPoint(x=float(pt[0]), y=float(pt[1]), size=10) for pt in kps1]
    kp2_cv = [cv2.KeyPoint(x=float(pt[0]), y=float(pt[1]), size=10) for pt in kps2]
    
    out_img = cv2.drawMatches(img1, kp1_cv, img2, kp2_cv, matches, None, 
                             flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    cv2.imwrite(save_path, out_img)

def validate(model, val_dataset, device, epoch, save_dir, log_file, train_config, mode):
    """
    验证函数:评估模型在真实数据集上的表现
    使用与 test_on_real.py 相同的评估流程
    """
    from measurement import calculate_metrics
    
    model.eval()
    all_metrics = []
    
    # 从配置中统一读取阈值
    nms_thresh = train_config.get('nms_thresh', 0.01)
    content_thresh = train_config.get('content_thresh', 0.7) # Lowe's Ratio
    geometric_thresh = train_config.get('geometric_thresh', 0.7) # RANSAC re-projection error
    
    epoch_save_dir = os.path.join(save_dir, f'epoch{epoch}')
    os.makedirs(epoch_save_dir, exist_ok=True)
    
    log_f = open(log_file, 'a')
    log_f.write(f'\n--- Validation Epoch {epoch} ---\n')
    
    # 图像预处理
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])
    
    with torch.no_grad():
        for i in tqdm(range(len(val_dataset)), desc=f"Val Epoch {epoch}"):
            raw_data = val_dataset.get_raw_sample(i)
            
            # 根据不同模态解包数据
            if mode == 'cfocta': # (cf, octa, pts_cf, pts_octa, path_cf, path_octa)
                img_fix_raw, img_mov_raw, pts_fix_gt, pts_mov_gt, path_fix, path_mov = raw_data
            elif mode == 'cffa': # (fa, cf, pts_fa, pts_cf, path_fa, path_cf)
                img_mov_raw, img_fix_raw, pts_mov_gt, pts_fix_gt, path_mov, path_fix = raw_data
            elif mode == 'cfoct': # (cf, oct, pts_cf, pts_oct, path_cf, path_oct)
                img_fix_raw, img_mov_raw, pts_fix_gt, pts_mov_gt, path_fix, path_mov = raw_data
            elif mode == 'octfa': # (fa, oct, pts_fa, pts_oct, path_fa, path_oct)
                img_mov_raw, img_fix_raw, pts_mov_gt, pts_fix_gt, path_mov, path_fix = raw_data
            
            # 确保灰度图用于模型输入
            if img_fix_raw.ndim == 3:
                img_fix_gray = cv2.cvtColor(img_fix_raw, cv2.COLOR_RGB2GRAY) if img_fix_raw.shape[2] == 3 else img_fix_raw.squeeze()
            else:
                img_fix_gray = img_fix_raw

            if img_mov_raw.ndim == 3:
                img_mov_gray = cv2.cvtColor(img_mov_raw, cv2.COLOR_RGB2GRAY) if img_mov_raw.shape[2] == 3 else img_mov_raw.squeeze()
            else:
                img_mov_gray = img_mov_raw
            
            sample_id = os.path.basename(path_fix).split('.')[0]
            
            # 准备模型输入
            img0_tensor = transform(img_fix_gray).unsqueeze(0).to(device)
            img1_tensor = transform(img_mov_gray).unsqueeze(0).to(device)
            
            # v4.1 Logic REMOVED for v6: 验证阶段保持原始输入，不反色
            # v6 模型 (Fix Encoder) 已经设计为直接处理暗血管特征
            img0_input = img0_tensor
            
            # 提取跨模态特征
            det_fix, desc_fix = model.network(img0_input, mode='fix')
            det_mov, desc_mov = model.network(img1_tensor, mode='mov')
            
            # 有效区域屏蔽,防止边缘伪影干扰关键点提取
            valid_mask = (img1_tensor > 0.05).float()
            valid_mask = -F.max_pool2d(-valid_mask, kernel_size=5, stride=1, padding=2)
            det_mov_masked = det_mov * valid_mask
            
            # 提取关键点
            kps_fix = nms(det_fix, nms_thresh=nms_thresh, nms_size=5)[0]
            kps_mov = nms(det_mov_masked, nms_thresh=nms_thresh, nms_size=5)[0]
            
            # 兜底策略:如果关键点太少,强制取响应最高的前100个点
            if len(kps_fix) < 10:
                flat_det = det_fix[0, 0].view(-1)
                _, idx = torch.topk(flat_det, min(100, flat_det.numel()))
                y = idx // det_fix.shape[3]; x = idx % det_fix.shape[3]
                kps_fix = torch.stack([x, y], dim=1).float()

            if len(kps_mov) < 10:
                flat_det = det_mov_masked[0, 0].view(-1)
                if flat_det.max() > 0:
                    _, idx = torch.topk(flat_det, min(100, flat_det.numel()))
                    y = idx // det_mov.shape[3]; x = idx % det_mov.shape[3]
                    kps_mov = torch.stack([x, y], dim=1).float()
            
            good_matches = []
            if len(kps_fix) >= 4 and len(kps_mov) >= 4:
                # 采样描述子并进行特征匹配
                desc_fix_samp = sample_keypoint_desc(kps_fix[None], desc_fix, s=8)[0]
                desc_mov_samp = sample_keypoint_desc(kps_mov[None], desc_mov, s=8)[0]
                
                d1 = desc_fix_samp.permute(1, 0).cpu().numpy()
                d2 = desc_mov_samp.permute(1, 0).cpu().numpy()
                
                matches = cv2.BFMatcher().knnMatch(d1, d2, k=2)
                
                # 使用配置中的 content_thresh 作为 Ratio Test 阈值
                for m, n in matches:
                    if m.distance < content_thresh * n.distance:
                        good_matches.append(m)
            
            # 映射回原始尺度
            h_f, w_f = img_fix_raw.shape[:2]
            h_m, w_m = img_mov_raw.shape[:2]
            
            kps_f_orig = kps_fix.cpu().numpy() * [w_f / 512.0, h_f / 512.0]
            kps_m_orig = kps_mov.cpu().numpy() * [w_m / 512.0, h_m / 512.0]
            
            mkpts0 = np.array([kps_f_orig[m.queryIdx] for m in good_matches]) if good_matches else np.array([])
            mkpts1 = np.array([kps_m_orig[m.trainIdx] for m in good_matches]) if good_matches else np.array([])
            
            # 为了统一评估，将 moving 图像 resize 到和 fix 相同尺寸
            # 这样匹配点和控制点都在同一尺寸空间中
            if (h_m, w_m) != (h_f, w_f):
                # Resize moving 图像到 fix 尺寸
                img_mov_resized = cv2.resize(img_mov_raw, (w_f, h_f), interpolation=cv2.INTER_LINEAR)
                
                # 调整 moving 侧的关键点和匹配点坐标
                scale_x = w_f / w_m
                scale_y = h_f / h_m
                
                # 只对非空数组进行缩放操作
                if len(mkpts1) > 0:
                    mkpts1 = mkpts1 * [scale_x, scale_y]
                if len(kps_m_orig) > 0:
                    kps_m_orig = kps_m_orig * [scale_x, scale_y]
                if len(pts_mov_gt) > 0:
                    pts_mov_gt = pts_mov_gt * [scale_x, scale_y]
            else:
                img_mov_resized = img_mov_raw
            
            # 计算 GT 单应矩阵
            if len(pts_fix_gt) >= 4:
                H_gt, _ = cv2.findHomography(pts_mov_gt, pts_fix_gt, cv2.RANSAC, 5.0)
            else:
                H_gt = None

            # 计算预测单应矩阵
            H_pred = None
            if len(mkpts0) >= 4:
                H_pred, _ = cv2.findHomography(mkpts1, mkpts0, cv2.RANSAC, geometric_thresh)
            
            # 最终估计矩阵: 成功则用 H_pred, 失败则用单位阵 (不注册)
            H_est = H_pred if H_pred is not None else np.eye(3)
            
            # 计算角点误差 MACE
            mace = compute_corner_error(H_est, H_gt, h_f, w_f) if H_gt is not None else float('inf')

            # 使用 measurement.py 获取其他指标 (Rep, MIR)
            metrics = calculate_metrics(
                img_origin=img_fix_raw, img_result=img_mov_resized,
                mkpts0=mkpts0, mkpts1=mkpts1,
                kpts0=kps_f_orig, kpts1=kps_m_orig,
                ctrl_pts0=pts_fix_gt, ctrl_pts1=pts_mov_gt,
                H_gt=H_gt
            )
            # 覆盖 mean_error 为 MACE
            metrics['mean_error'] = mace
            
            all_metrics.append(metrics)
            
            log_f.write(f"ID: {sample_id} | SR_ME: {metrics['SR_ME']} | SR_MAE: {metrics['SR_MAE']} | "
                       f"Rep: {metrics['Rep']:.4f} | MIR: {metrics['MIR']:.4f} | "
                       f"MACE: {metrics['mean_error']:.2f} px\n")
            
            # 保存可视化结果
            sample_save_dir = os.path.join(epoch_save_dir, sample_id)
            os.makedirs(sample_save_dir, exist_ok=True)
            
            # 保存原始图像（moving 使用统一尺寸后的版本）
            cv2.imwrite(os.path.join(sample_save_dir, f'{sample_id}_fix.png'), img_fix_raw)
            cv2.imwrite(os.path.join(sample_save_dir, f'{sample_id}_moving.png'), img_mov_resized)

            # 绘制并保存带有关键点的图像 (Fix & Moving)
            kp_f_cv = [cv2.KeyPoint(x=float(pt[0]), y=float(pt[1]), size=10) for pt in kps_f_orig]
            # 注意: drawKeypoints 会自动处理灰度/彩色输入，返回彩色图像
            img_fix_kpts = cv2.drawKeypoints(img_fix_raw, kp_f_cv, None, color=(0, 255, 0), flags=0)
            
            # --- DEBUG INFO: 在图片上打印调试信息 ---
            # 统计 Heatmap 极值，判断是否全黑或响应过低
            det_max = det_fix.max().item()
            det_mean = det_fix.mean().item()
            txt = f"Det Max: {det_max:.4f} Mean: {det_mean:.4f} Pts: {len(kps_f_orig)}"
            cv2.putText(img_fix_kpts, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            
            # 统计 GT Mask 信息 (如果有) - 既然是 val，我们可以尝试从 memory 或 disk 再次验证 mask 状态
            # 这里简单打印一下本次检测到的点坐标范围，看是否集中在 (0,0)
            if len(kps_f_orig) > 0:
                min_x, max_x = kps_f_orig[:, 0].min(), kps_f_orig[:, 0].max()
                min_y, max_y = kps_f_orig[:, 1].min(), kps_f_orig[:, 1].max()
                t2 = f"X: {min_x:.1f}-{max_x:.1f} Y: {min_y:.1f}-{max_y:.1f}"
                cv2.putText(img_fix_kpts, t2, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            cv2.imwrite(os.path.join(sample_save_dir, f'{sample_id}_fix_kpts.png'), img_fix_kpts)

            kp_m_cv = [cv2.KeyPoint(x=float(pt[0]), y=float(pt[1]), size=10) for pt in kps_m_orig]
            img_mov_kpts = cv2.drawKeypoints(img_mov_resized, kp_m_cv, None, color=(0, 255, 0), flags=0)
            
            # --- DEBUG INFO MOVING ---
            det_m_max = det_mov.max().item()
            det_m_mean = det_mov.mean().item()
            valid_area_ratio = valid_mask.mean().item()
            txt_m = f"Det Max: {det_m_max:.4f} Mean: {det_m_mean:.4f} Pts: {len(kps_m_orig)}"
            cv2.putText(img_mov_kpts, txt_m, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            
            t2_m = f"Msk Ratio: {valid_area_ratio:.2f}"
            cv2.putText(img_mov_kpts, t2_m, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            
            cv2.imwrite(os.path.join(sample_save_dir, f'{sample_id}_moving_kpts.png'), img_mov_kpts)

            # 无论是否有足够的匹配点，都进行可视化保存
            # 确保图像为灰度图（用于配准和棋盘格）
            img_m_gray = cv2.cvtColor(img_mov_resized, cv2.COLOR_RGB2GRAY) if img_mov_resized.ndim == 3 else img_mov_resized
            img_f_gray = cv2.cvtColor(img_fix_raw, cv2.COLOR_RGB2GRAY) if img_fix_raw.ndim == 3 else img_fix_raw
            
            # 使用 H_est (可能是单位阵) 配准并保存结果
            reg_img = cv2.warpPerspective(img_m_gray, H_est, (w_f, h_f))
            cv2.imwrite(os.path.join(sample_save_dir, f'{sample_id}_moving_result.png'), reg_img)
            
            # 记录棋盘格
            checker = compute_checkerboard(img_f_gray, reg_img, n_grid=4)
            cv2.imwrite(os.path.join(sample_save_dir, f'{sample_id}_checkerboard.png'), checker)
            
            # 绘制匹配关系（使用统一尺寸后的图像）
            draw_matches(img_fix_raw, kps_f_orig, img_mov_resized, kps_m_orig, good_matches, 
                        os.path.join(sample_save_dir, f'{sample_id}_matches.png'))

    # 计算平均指标 (包含失败样本)
    summary = {}
    for key in ['SR_ME', 'SR_MAE', 'Rep', 'MIR', 'mean_error', 'max_error']:
        vals = [m[key] for m in all_metrics]
        summary[key] = np.mean(vals) if vals else 0.0
    
    # 计算 AUC@10 (Area Under Curve of Cumulative Error Distribution up to 10px)
    errors = np.array([m['mean_error'] for m in all_metrics])
    # 处理 inf 误差 (配准失败)
    errors[np.isinf(errors)] = 1e9 
    
    thresholds = np.linspace(0, 10, 1000) # 0到10px，1000个采样点
    # 计算每个阈值下的累积成功率 (CDF)
    success_rates = [np.mean(errors <= t) for t in thresholds]
    # 计算曲线下面积 (归一化到 0-1 范围，本来积分是 0-10，除以10做归一化)
    # 实际上 AUC@10 通常指 success rate 的平均值，或者积分值。这里归一化到 0-1 更直观。
    auc_10 = np.trapz(success_rates, thresholds) / 10.0
    
    log_f.write(f"\n--- Validation Summary ---\n")
    log_f.write(f"Overall SR_ME (Success Rate @5px):  {summary['SR_ME']*100:.2f}%\n")
    log_f.write(f"Overall SR_MAE (Success Rate @10px): {summary['SR_MAE']*100:.2f}%\n")
    log_f.write(f"AUC@10:                             {auc_10:.4f}\n")
    log_f.write(f"Average Repeatability:              {summary['Rep']*100:.2f}%\n")
    log_f.write(f"Average Matching Inliers Ratio:     {summary['MIR']*100:.2f}%\n")
    log_f.write(f"Overall MACE (Mean Corner Error):   {summary['mean_error']:.2f} px\n")
    log_f.write(f"Max Registration Error (Average):   {summary['max_error']:.2f} px\n")
    log_f.close()
    
    print(f'Validation Epoch {epoch} Finished. AUC@10: {auc_10:.4f}, MACE: {summary["mean_error"]:.2f} px')
    return auc_10

def train_multimodal():
    """
    多模态训练主流程 - 使用新数据集和三阶段课程学习
    """
    # 加载配置
    config_path = './config/train_multimodal.yaml'
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
        
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Command line args to override config
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', '-n', type=str, help='Experiment name', default=None)
    parser.add_argument('--mode', '-m', type=str, choices=['cffa', 'cfoct', 'octfa', 'cfocta'], 
                        help='Registration mode', default=None)
    parser.add_argument('--epoch', '-e', type=int, help='Number of training epochs', default=500)
    parser.add_argument('--batch_size', '-b', type=int, help='Batch size for training', default=4)
    parser.add_argument('--geometric_thresh', '-g', type=float, help='RANSAC geometric threshold for PKE', default=0.7)
    parser.add_argument('--content_thresh', '-c', type=float, help='Lowe ratio threshold for feature matching', default=0.8)
    args = parser.parse_args()
    
    if args.name:
        config['MODEL']['name'] = args.name
    if args.mode:
        config['DATASET']['registration_type'] = args.mode
    if args.epoch:
        config['MODEL']['num_epoch'] = args.epoch
    if args.batch_size:
        config['DATASET']['batch_size'] = args.batch_size
    if args.geometric_thresh is not None:
        config['PKE']['geometric_thresh'] = args.geometric_thresh
    if args.content_thresh is not None:
        config['PKE']['content_thresh'] = args.content_thresh
        
    # v6: Dual-Path Encoder (False)
    config['MODEL']['shared_encoder'] = False
        
    train_config = {**config['MODEL'], **config['PKE'], **config['DATASET'], **config['VALUE_MAP']}
    
    exp_name = train_config.get('name', 'default_exp')
    reg_type = train_config['registration_type']
    save_root = f'./save/{reg_type}/{exp_name}'
    os.makedirs(save_root, exist_ok=True)
    
    log_file = os.path.join(save_root, 'validation_log.txt')
    train_log_file = os.path.join(save_root, 'train_log.txt')  # 新增：训练日志
    
    device = torch.device(train_config['device'] if torch.cuda.is_available() else "cpu")
    
    # 打开训练日志文件
    train_log = open(train_log_file, 'a', buffering=1)  # 行缓冲，实时写入
    
    def log_print(msg):
        """同时输出到控制台和日志文件"""
        print(msg)
        train_log.write(msg + '\n')
        train_log.flush()
    
    log_print(f"Using device: {device} | Experiment: {exp_name}")

    # 数据加载 - 使用新的FIVES数据集进行训练
    root_dir = train_config['root_dir']
    batch_size = train_config['batch_size']
    img_size = train_config.get('img_size', 512)
    df = train_config.get('df', 8)
    
    train_set = MultiModalDataset(
        root_dir=root_dir, 
        mode=reg_type, 
        split='train', 
        img_size=img_size, 
        df=df
    )
    
    # 验证集使用真实数据集 (与 test_on_real.py 一致)
    if reg_type == 'cfocta':
        val_set = CFOCTADataset(root_dir='dataset/CF_OCTA_v2_repaired', split='val', mode='cf2octa')
    elif reg_type == 'cffa':
        val_set = CFFADataset(root_dir='dataset/operation_pre_filtered_cffa', split='val', mode='fa2cf')
    elif reg_type == 'cfoct':
        val_set = CFOCTDataset(root_dir='dataset/operation_pre_filtered_cfoct', split='val', mode='cf2oct')
    elif reg_type == 'octfa':
        val_set = OCTFADataset(root_dir='dataset/operation_pre_filtered_octfa', split='val', mode='fa2oct')
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    
    # 初始化多模态 SuperRetina 模型
    model = SuperRetinaMultimodal(train_config, device=device)
    
    if train_config['load_pre_trained_model']:
        path = train_config['pretrained_path']
        if os.path.exists(path):
            log_print(f"Loading pretrained model from {path}")
            checkpoint = torch.load(path, map_location=device)
            model.load_state_dict(checkpoint['net'])
            
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    num_epochs = train_config['num_epoch']
    pke_start_epoch = train_config['pke_start_epoch']
    
    is_value_map_save = train_config['is_value_map_save']
    value_map_save_dir = train_config['value_map_save_dir']
    
    if is_value_map_save:
        if os.path.exists(value_map_save_dir):
            shutil.rmtree(value_map_save_dir)
        os.makedirs(value_map_save_dir)
        
    value_maps_running = {} if not is_value_map_save else None
    
    # 最佳指标追踪 (AUC@10 越大越好)
    best_auc = 0.0
    
    # 早停机制变量 (仅在epoch >= 100后启用)
    patience = 5  # 验证指标连续5次不提升则早停
    patience_counter = 0
    best_val_auc = 0.0

    # 初始验证
    log_print("Running initial validation...")
    _ = validate(model, val_set, device, 0, save_root, log_file, train_config, reg_type)

    # v6: 全程启用 GT-Init PKE
    pke_start_epoch = 0 
    
    # ... (the for loop starts here)
    # I'll rely on chunk matching. I am replacing the variable init part.
    # And then I need to update the save logic inside the loop.
    # It is safer to update the VARIABLE INIT here, and then update the LOOP BODY in a separate call or same call if contiguous.
    # They are separated by `for epoch in range...` which is large.
    # Let's fix the init variables first.
    
    # Wait, replace_file_content replaces a CONTIGUOUS block. 
    # The variable init and the saving logic (at end of loop) are far apart.
    # I should do this in two steps or use MultiReplace (available).
    # I will use replace_file_content for the Init first.
    
    # Actually, I'll use MultiReplaceFileContent to do both at once if possible, or just two calls.
    # Let's use two calls to be safe and simple.
    
    # First call: Variable Init.


    # 初始验证
    log_print("Running initial validation...")
    _ = validate(model, val_set, device, 0, save_root, log_file, train_config, reg_type)

    # v6: 全程启用 GT-Init PKE
    pke_start_epoch = 0 
    
    # 训练循环
    for epoch in range(1, num_epochs + 1):
        # v6: 统一进入自监督+GT引导模式
        phase = 3 
        pke_supervised = False 
        model.PKE_learn = True
        phase_name = "v6: Dual-Path PKE + GT Anchor + Mask Constraint"
            
        log_print(f'Epoch {epoch}/{num_epochs} | {phase_name}')
        model.train()
            
        running_loss_det = 0.0
        running_loss_desc = 0.0
        total_samples = 0
        
        for step_idx, data in enumerate(tqdm(train_loader, desc=f"Train Epoch {epoch}")):
            img0_orig = data['image0'].to(device)
            img1_orig = data['image1'].to(device)
            
            # 域随机化
            img0 = apply_domain_randomization(img0_orig)
            img1 = apply_domain_randomization(img1_orig)
            
            # v6: 移除反色 (No Inversion), 直接使用原始强度
            img0_input = img0
            
            # ===== 可视化: 保存第一个 epoch 的前两个 batch =====
            if epoch == 1 and step_idx < 2:
                save_batch_visualization(
                    img0_orig, img1_orig, img0_input, img1,
                    save_root, epoch, step_idx + 1, batch_size,
                    vessel_mask=data['vessel_mask0'].to(device)
                )
            # ===== 关键修改：从完整血管分割图中提取稀疏的分叉点 =====
            # 这些分叉点将作为训练时的监督信号，引导模型学习独特的关键点
            vessel_mask_full = data['vessel_mask0'].to(device)  # [B, 1, H, W]
            vessel_keypoints_batch = []
            
            for b in range(vessel_mask_full.shape[0]):
                # 转换为 numpy 格式 (H, W) - 修复数值溢出问题
                mask_tensor = vessel_mask_full[b, 0].cpu()
                if mask_tensor.max() <= 1.0:
                    mask_np = (mask_tensor.numpy() * 255).astype(np.uint8)
                else:
                    mask_np = mask_tensor.numpy().astype(np.uint8)
                
                # 提取关键点
                try:
                    keypoints = extract_vessel_keypoints(mask_np, min_distance=8)
                except:
                    try:
                        keypoints = extract_vessel_keypoints_fallback(mask_np, min_distance=8)
                    except:
                        # 如果提取失败，使用原始掩码
                        keypoints = (mask_np > 127).astype(np.float32)
                        print("点位提取失败")
                
                # 调试: 检查点位数量 (打印前两个 batch 的采样情况)
                if step_idx < 2:
                    print(f"Sample {b}: Found {np.sum(keypoints)} vessel keypoints")
                    
                vessel_keypoints_batch.append(torch.from_numpy(keypoints).float())

            # 转换回 tensor [B, 1, H, W]
            vessel_keypoints = torch.stack(vessel_keypoints_batch, dim=0).unsqueeze(1).to(device)
            
            
            # 准备 PKE 训练所需参数
            batch_size = img0.size(0)
            input_with_labels = torch.ones(batch_size, dtype=torch.bool).to(device)
            learn_index = torch.where(input_with_labels)
            
            # 加载动态 Value Maps (记录每个像素点的历史置信度)
            names = data['pair_names'][0] # 使用固定图名称作为 key
            value_maps = value_map_load(value_map_save_dir, names, input_with_labels, 
                                      img0.shape[-2:], value_maps_running)
            value_maps = value_maps.to(device)
            
            # 读取真值几何变换矩阵 H_0to1 (image0 -> image1)，用于描述子热身阶段
            H_0to1 = data.get('T_0to1', None)
            if H_0to1 is not None:
                H_0to1 = H_0to1.to(device)
            
            optimizer.zero_grad()
            
            with torch.set_grad_enabled(True):
                # 调用模型 forward 方法 - v6 Interface
                # 传入 vessel_keypoints 作为 GT Anchor
                # 传入 vessel_mask_full 用于 Mask Constraint
                loss, number_pts, loss_det_item, loss_desc_item, enhanced_kp, enhanced_label, det_pred, n_det, n_desc = \
                    model(img0_input, img1, vessel_keypoints, value_maps, learn_index,
                          phase=phase, vessel_mask=vessel_mask_full, H_0to1=H_0to1,
                          pke_supervised=pke_supervised) # model.forward v6 will remove vessel_weight
                    
                loss.backward()
                optimizer.step()
                
            # 更新持久化的 Value Maps
            if len(learn_index[0]) > 0:
                value_maps = value_maps.cpu()
                value_map_save(value_map_save_dir, names, input_with_labels, value_maps, value_maps_running)
                    
            running_loss_det += loss_det_item
            running_loss_desc += loss_desc_item
            total_samples += img0.size(0)

        epoch_loss = (running_loss_det + running_loss_desc) / total_samples
        log_print(f'Train Total Loss: {epoch_loss:.4f} (Det: {running_loss_det/total_samples:.4f}, Desc: {running_loss_desc/total_samples:.4f})')
        
        # 每 5 个 Epoch 进行一次验证并保存模型
        if epoch % 5 == 0:
            auc_test = validate(model, val_set, device, epoch, save_root, log_file, train_config, reg_type)
            
            state = {
                'net': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'auc': auc_test
            }
            
            # 保存最新模型
            latest_dir = os.path.join(save_root, 'latestpoint')
            os.makedirs(latest_dir, exist_ok=True)
            torch.save(state, os.path.join(latest_dir, 'checkpoint.pth'))
            # 保存epoch信息
            with open(os.path.join(latest_dir, 'checkpoint_info.txt'), 'w') as f:
                f.write(f'Latest Checkpoint\nEpoch: {epoch}\nAUC@10: {auc_test:.4f}\n')
            
            # 保存 AUC 表现最好的模型 (越大越好)
            if auc_test > best_auc:
                log_print(f"New Best AUC: {auc_test:.4f} (Previous: {best_auc:.4f})")
                best_auc = auc_test
                best_dir = os.path.join(save_root, 'bestcheckpoint')
                os.makedirs(best_dir, exist_ok=True)
                torch.save(state, os.path.join(best_dir, 'checkpoint.pth'))
                # 保存epoch信息
                with open(os.path.join(best_dir, 'checkpoint_info.txt'), 'w') as f:
                    f.write(f'Best Checkpoint\nEpoch: {epoch}\nAUC@10: {auc_test:.4f}\n')
            
            # 早停机制 (仅在 epoch >= 100 后启用)
            if epoch >= 100:
                if auc_test > best_val_auc:
                    best_val_auc = auc_test
                    patience_counter = 0
                    log_print(f'[Early Stopping] Validation AUC improved to {best_val_auc:.4f}. Reset patience counter.')
                else:
                    patience_counter += 1
                    log_print(f'[Early Stopping] Validation AUC did not improve. Patience: {patience_counter}/{patience}')
                
                if patience_counter >= patience:
                    log_print(f'Early stopping triggered at epoch {epoch}. Best validation AUC: {best_val_auc:.4f}')
                    break
    
    # 训练结束，关闭日志文件
    train_log.close()

if __name__ == '__main__':
    train_multimodal()
