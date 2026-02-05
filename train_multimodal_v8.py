"""
SuperRetina v8: CF-Anchor 多模态对齐训练脚本
目标：取消 PKE，以 CF 为锚点，通过密集描述子对齐和检测头蒸馏训练 Moving Encoder。
"""

import torch
import os
import sys
import yaml
import shutil
import cv2
import numpy as np
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import argparse
from torchvision import transforms

# 添加本地模块路径
sys.path.append(os.getcwd())

# 导入必要模块
from dataset.FIVES_extract_v3.FIVES_extract_v3 import MultiModalDataset
from dataset.CF_OCTA_v2_repaired.cf_octa_v2_repaired_dataset import CFOCTADataset
from dataset.operation_pre_filtered_cffa.operation_pre_filtered_cffa_dataset import CFFADataset
from dataset.operation_pre_filtered_cfoct.operation_pre_filtered_cfoct_dataset import CFOCTDataset
from dataset.operation_pre_filtered_octfa.operation_pre_filtered_octfa_dataset import OCTFADataset
from model.super_retina_multimodal import SuperRetinaMultimodal
from common.train_util import affine_images
from common.common_util import nms, sample_keypoint_desc
from gen_data_enhance_v2 import apply_domain_randomization, save_batch_visualization

# ============= 工具函数 (保留验证逻辑) =============

def compute_corner_error(H_est, H_gt, height, width):
    corners = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
    corners_homo = np.concatenate([corners, np.ones((4, 1), dtype=np.float32)], axis=1)
    if H_gt is None: return float('inf')
    corners_gt_homo = (H_gt @ corners_homo.T).T
    corners_gt = corners_gt_homo[:, :2] / (corners_gt_homo[:, 2:] + 1e-6)
    if H_est is None or not np.isfinite(H_est).all(): return float('inf')
    try:
        corners_est_homo = (H_est @ corners_homo.T).T
        corners_est = corners_est_homo[:, :2] / (corners_est_homo[:, 2:] + 1e-6)
        return np.mean(np.sqrt(np.sum((corners_est - corners_gt)**2, axis=1)))
    except: return float('inf')

def compute_checkerboard(img1, img2, n_grid=4):
    h, w = img1.shape[:2]
    grid_h, grid_w = h // n_grid, w // n_grid
    checkerboard = np.zeros_like(img1)
    for i in range(n_grid):
        for j in range(n_grid):
            y_s, y_e = i * grid_h, (i + 1) * grid_h if i < n_grid - 1 else h
            x_s, x_e = j * grid_w, (j + 1) * grid_w if j < n_grid - 1 else w
            checkerboard[y_s:y_e, x_s:x_e] = img1[y_s:y_e, x_s:x_e] if (i + j) % 2 == 0 else img2[y_s:y_e, x_s:x_e]
    return checkerboard

def visualize_descriptors_pca(desc_fix, desc_mov):
    B, C, H, W = desc_fix.shape
    feat_fix = desc_fix[0].view(C, -1).permute(1, 0)
    feat_mov = desc_mov[0].view(C, -1).permute(1, 0)
    combined = torch.cat([feat_fix, feat_mov], dim=0)
    combined = F.normalize(combined, dim=-1)
    try:
        _, _, V = torch.pca_lowrank(combined, q=3, niter=2)
        pca_feat = (torch.mm(combined, V) - torch.mm(combined, V).min()) / (torch.mm(combined, V).max() - torch.mm(combined, V).min() + 1e-8)
        return (pca_feat[:H*W].view(H, W, 3).cpu().numpy() * 255).astype(np.uint8), \
               (pca_feat[H*W:].view(H, W, 3).cpu().numpy() * 255).astype(np.uint8)
    except: return np.zeros((H, W, 3), np.uint8), np.zeros((H, W, 3), np.uint8)

def validate(model, val_dataset, device, epoch, save_dir, log_file, train_config, mode):
    """
    验证函数: 评估模型在真实数据集上的表现 (对齐论文评估标准)
    """
    from measurement_SuperRetina import calculate_metrics, compute_auc
    model.eval()
    all_metrics = []
    
    epoch_save_dir = os.path.join(save_dir, f'epoch{epoch}')
    os.makedirs(epoch_save_dir, exist_ok=True)
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])
    
    with torch.no_grad():
        for i in tqdm(range(len(val_dataset)), desc=f"Val Epoch {epoch}"):
            raw_data = val_dataset.get_raw_sample(i)
            # 根据模态解包数据
            if mode == 'cfocta': img_fix_raw, img_mov_raw, pts_fix_gt, pts_mov_gt, path_fix, path_mov = raw_data
            elif mode == 'cffa': img_mov_raw, img_fix_raw, pts_mov_gt, pts_fix_gt, path_mov, path_fix = raw_data
            elif mode == 'cfoct': img_fix_raw, img_mov_raw, pts_fix_gt, pts_mov_gt, path_fix, path_mov = raw_data
            else: img_mov_raw, img_fix_raw, pts_mov_gt, pts_fix_gt, path_mov, path_fix = raw_data
            
            img_fix_gray = cv2.cvtColor(img_fix_raw, cv2.COLOR_RGB2GRAY) if img_fix_raw.ndim == 3 else img_fix_raw
            img_mov_gray = cv2.cvtColor(img_mov_raw, cv2.COLOR_RGB2GRAY) if img_mov_raw.ndim == 3 else img_mov_raw
            
            img0_tensor = transform(img_fix_gray).unsqueeze(0).to(device)
            img1_tensor = transform(img_mov_gray).unsqueeze(0).to(device)
            
            det_fix, desc_fix = model.network(img0_tensor, mode='fix')
            det_mov, desc_mov = model.network(img1_tensor, mode='mov')
            
            # --- 可视化 (PCA & Matches) ---
            sample_id = os.path.basename(path_fix).split('.')[0]
            sample_save_dir = os.path.join(epoch_save_dir, sample_id)
            os.makedirs(sample_save_dir, exist_ok=True)
            
            pca_fix, pca_mov = visualize_descriptors_pca(desc_fix, desc_mov)
            cv2.imwrite(os.path.join(sample_save_dir, 'pca_fix.png'), cv2.cvtColor(pca_fix, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(sample_save_dir, 'pca_mov.png'), cv2.cvtColor(pca_mov, cv2.COLOR_RGB2BGR))
            
            # 关键点提取与匹配
            kps_fix = nms(det_fix, nms_thresh=1e-6, nms_size=5)[0]
            kps_mov = nms(det_mov, nms_thresh=1e-6, nms_size=5)[0]
            
            # Top-300 点截断
            if len(kps_fix) > 300:
                scores_f = det_fix[0, 0, kps_fix[:, 1].long(), kps_fix[:, 0].long()]
                _, idx_f = torch.topk(scores_f, 300)
                kps_fix = kps_fix[idx_f]
            if len(kps_mov) > 300:
                scores_m = det_mov[0, 0, kps_mov[:, 1].long(), kps_mov[:, 0].long()]
                _, idx_m = torch.topk(scores_m, 300)
                kps_mov = kps_mov[idx_m]
                
            good_matches = []
            H_est = np.eye(3)
            if len(kps_fix) >= 4 and len(kps_mov) >= 4:
                desc_f = sample_keypoint_desc(kps_fix[None], desc_fix, s=8)[0].permute(1, 0).cpu().numpy()
                desc_m = sample_keypoint_desc(kps_mov[None], desc_mov, s=8)[0].permute(1, 0).cpu().numpy()
                matches = cv2.BFMatcher().knnMatch(desc_f, desc_m, k=2)
                good_matches = [m for m, n in matches if m.distance < 0.8 * n.distance]
                
                if len(good_matches) >= 4:
                    h_f, w_f = img_fix_raw.shape[:2]
                    h_m, w_m = img_mov_raw.shape[:2]
                    mkpts0 = kps_fix.cpu().numpy()[ [m.queryIdx for m in good_matches] ] * [w_f/512, h_f/512]
                    mkpts1 = kps_mov.cpu().numpy()[ [m.trainIdx for m in good_matches] ] * [w_m/512, h_m/512]
                    H_p, _ = cv2.findHomography(mkpts1, mkpts0, cv2.RANSAC, 5.0)
                    if H_p is not None: H_est = H_p

            # 计算指标 (使用 measurement_SuperRetina)
            orig_size_ref = (2912, 2912)
            pts_f_p = pts_fix_gt * [orig_size_ref[1]/img_fix_raw.shape[1], orig_size_ref[0]/img_fix_raw.shape[0]]
            pts_m_p = pts_mov_gt * [orig_size_ref[1]/img_mov_raw.shape[1], orig_size_ref[0]/img_mov_raw.shape[0]]
            
            # 缩放后的预测匹配点
            mkpts0_p = mkpts0 * [orig_size_ref[1]/w_f, orig_size_ref[0]/h_f] if len(good_matches) >= 4 else np.array([])
            mkpts1_p = mkpts1 * [orig_size_ref[1]/w_m, orig_size_ref[0]/h_m] if len(good_matches) >= 4 else np.array([])
            
            metrics = calculate_metrics(mkpts0_p, mkpts1_p, pts_f_p, pts_m_p, orig_size=orig_size_ref)
            all_metrics.append(metrics)
            
            # 保存棋盘格可视化
            reg_img = cv2.warpPerspective(img_mov_gray, H_est, (img_fix_raw.shape[1], img_fix_raw.shape[0]))
            checker = compute_checkerboard(img_fix_gray, reg_img, n_grid=4)
            cv2.imwrite(os.path.join(sample_save_dir, 'checkerboard.png'), checker)

    # 汇总
    acc_rate = np.mean([m['is_acceptable'] for m in all_metrics])
    errors = []
    for m in all_metrics: errors.extend(m['errors'])
    auc20 = compute_auc(errors, max_threshold=20)
    
    print(f"Epoch {epoch} | Acc Rate: {acc_rate*100:.2f}% | AUC@20: {auc20:.4f}")
    if log_file:
        with open(log_file, 'a') as f:
            f.write(f"Epoch {epoch} | Acc: {acc_rate:.4f} | AUC20: {auc20:.4f}\n")
            
    return auc20

# ============= 训练主流程 (v8) =============

def train_v8():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', '-n', type=str, default='v8_anchor_alignment')
    parser.add_argument('--mode', '-m', type=str, default='cffa', choices=['cffa', 'cfoct', 'octfa', 'cfocta'])
    parser.add_argument('--epoch', '-e', type=int, default=500)
    args = parser.parse_args()

    # 1. 配置加载
    with open('./config/train_multimodal.yaml') as f:
        config = yaml.safe_load(f)
    
    train_config = {**config['MODEL'], **config['DATASET']}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_root = f'./save/{args.mode}/{args.name}'
    os.makedirs(save_root, exist_ok=True)
    
    # 2. 模型初始化与权重加载
    # 强制设置 shared_encoder 为 False 以使用双路结构
    train_config['shared_encoder'] = False
    model = SuperRetinaMultimodal(train_config, device=device)
    
    if os.path.exists('weights/SuperRetina.pth'):
        print(f"加载预训练权重并分发至双路编码器 (Initial Anchor)")
        ckpt = torch.load('weights/SuperRetina.pth', map_location=device)['net']
        model_dict = model.state_dict()
        new_dict = {}
        for k, v in ckpt.items():
            if k.startswith('encoder.'):
                new_dict[k.replace('encoder.', 'encoder_fix.')] = v
                new_dict[k.replace('encoder.', 'encoder_mov.')] = v
            elif any(k.startswith(p + '.') for p in ['conv1a', 'conv1b', 'conv2a', 'conv2b', 'conv3a', 'conv3b', 'conv4a', 'conv4b']):
                new_dict['encoder_fix.' + k] = v
                new_dict['encoder_mov.' + k] = v
            elif k in model_dict:
                new_dict[k] = v
        model.load_state_dict(new_dict, strict=False)

    # 3. 冻结策略 (v8 核心：冻结 CF 锚点)
    for n, p in model.named_parameters():
        if 'encoder_fix' in n:
            p.requires_grad = False
        else:
            p.requires_grad = True # 训练 encoder_mov 和所有 Head
            
    optimizer = optim.Adam([
        {'params': model.encoder_mov.parameters(), 'lr': 1e-5}, # Moving Encoder 低学习率
        {'params': [p for n, p in model.named_parameters() if 'encoder' not in n], 'lr': 1e-4} # Head 正常学习率
    ])

    # 4. 数据加载
    train_set = MultiModalDataset(root_dir=train_config['root_dir'], mode=args.mode, split='train', img_size=512)
    train_loader = DataLoader(train_set, batch_size=train_config['batch_size'], shuffle=True, num_workers=4)
    
    # 验证集初始化
    if args.mode == 'cfocta': val_set = CFOCTADataset(root_dir='dataset/CF_OCTA_v2_repaired', split='val', mode='cf2octa')
    elif args.mode == 'cffa': val_set = CFFADataset(root_dir='dataset/operation_pre_filtered_cffa', split='val', mode='fa2cf')
    
    # 5. 训练循环
    best_auc = -1.0
    for epoch in range(1, args.epoch + 1):
        model.train()
        total_loss, total_desc, total_det = 0, 0, 0
        
        for data in tqdm(train_loader, desc=f"Epoch {epoch}"):
            # v8 训练使用对齐良好的模拟数据对
            img_fix = apply_domain_randomization(data['image0'].to(device))
            img_mov = apply_domain_randomization(data['image1'].to(device))
            H_0to1 = data['T_0to1'].to(device)
            vessel_mask = data['vessel_mask0'].to(device)
            
            optimizer.zero_grad()
            
            # 调用 v8 特有的 forward 方法
            loss, l_desc, l_det, _, _ = model.forward_v8(
                img_fix, img_mov, H_0to1, vessel_mask,
                lambda_desc=1.0, lambda_det=0.5
            )
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_desc += l_desc.item()
            total_det += l_det.item()
            
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch} Loss: {avg_loss:.4f} (Desc: {total_desc/len(train_loader):.4f}, Det: {total_det/len(train_loader):.4f})")
        
        # 定期验证于保存
        if epoch % 10 == 0:
            auc = validate(model, val_set, device, epoch, save_root, None, train_config, args.mode)
            if auc > best_auc:
                best_auc = auc
                torch.save({'net': model.state_dict(), 'epoch': epoch, 'auc': auc}, os.path.join(save_root, 'best.pth'))
    
if __name__ == '__main__':
    train_v8()
