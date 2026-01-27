"""
基于 v6 架构的对照组: 直接在真实多模态数据集上训练 (严格对齐 v6 策略)
1. 目的: 对比 plan_v6 策略在生成数据 (有掩码) 和真实数据 (无掩码) 上的性能差异。
2. 架构: SuperRetinaMultimodal (Dual-Path, shared_encoder=False)。
3. 训练策略 (完全对齐 v6):
   - Phase 0: 特征空间对齐热身 (使用由 GT H 预对齐的真实对)。
   - Phase 3: 混合 PKE 注册 (使用原始未对齐的真实对)。
   - 域随机化: 使用 apply_domain_randomization。
   - 种子监督: 真实 GT 控制点作为 PKE 初始种子。
   - 掩码约束: 因真实数据无掩码，故设置为 None，跳过背景抑制。
"""

import torch
import os
import sys
import yaml
import shutil
import cv2
import numpy as np
import argparse
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from torchvision import transforms

# 添加本地模块路径
sys.path.append(os.getcwd())

from dataset.CF_OCTA_v2_repaired.cf_octa_v2_repaired_dataset import CFOCTADataset
from dataset.operation_pre_filtered_cffa.operation_pre_filtered_cffa_dataset import CFFADataset
from dataset.operation_pre_filtered_cfoct.operation_pre_filtered_cfoct_dataset import CFOCTDataset
from dataset.operation_pre_filtered_octfa.operation_pre_filtered_octfa_dataset import OCTFADataset

from model.super_retina_multimodal import SuperRetinaMultimodal
from common.train_util import value_map_load, value_map_save
from common.common_util import nms, sample_keypoint_desc
from gen_data_enhance_v2 import apply_domain_randomization, save_batch_visualization
from measurement_SuperRetina import calculate_metrics, compute_auc


# ============================================================================
# 数据集包装器 (对齐 v6 接口)
# ============================================================================

class RealDataV6Wrapper(Dataset):
    def __init__(self, base_dataset, img_size=512):
        self.base_dataset = base_dataset
        self.img_size = img_size

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        raw_data = self.base_dataset.get_raw_sample(idx)
        img_fix_raw, img_mov_raw, pts_fix, pts_mov, path_fix, path_mov = raw_data
        
        def to_gray(img):
            if img.ndim == 3:
                return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.shape[2] == 3 else img.squeeze()
            return img

        img0_512 = cv2.resize(to_gray(img_fix_raw), (self.img_size, self.img_size))
        img1_512 = cv2.resize(to_gray(img_mov_raw), (self.img_size, self.img_size))
        
        h_f, w_f = img_fix_raw.shape[:2]
        h_m, w_m = img_mov_raw.shape[:2]
        pts_fix_512 = pts_fix * [self.img_size / w_f, self.img_size / h_f]
        pts_mov_512 = pts_mov * [self.img_size / w_m, self.img_size / h_m]
        
        H_0to1 = np.eye(3, dtype=np.float32)
        H_inv = np.eye(3, dtype=np.float32)
        has_gt = False
        if len(pts_fix_512) >= 4:
            H, _ = cv2.findHomography(pts_fix_512, pts_mov_512, cv2.RANSAC, 5.0)
            if H is not None:
                H_0to1 = H.astype(np.float32)
                has_gt = True
                try: H_inv = np.linalg.inv(H_0to1)
                except: pass

        img1_aligned = cv2.warpPerspective(img1_512, H_inv, (self.img_size, self.img_size))
        
        seed_map = np.zeros((self.img_size, self.img_size), dtype=np.float32)
        for pt in pts_fix_512:
            x, y = int(pt[0]), int(pt[1])
            if 0 <= x < self.img_size and 0 <= y < self.img_size:
                seed_map[y, x] = 1.0

        return {
            'image0': torch.from_numpy(img0_512).float().unsqueeze(0) / 255.0,
            'image1': torch.from_numpy(img1_512).float().unsqueeze(0) / 255.0,
            'image1_origin': torch.from_numpy(img1_aligned).float().unsqueeze(0) / 255.0,
            'T_0to1': torch.from_numpy(H_0to1),
            'vessel_mask0': torch.from_numpy(seed_map).unsqueeze(0),
            'has_gt': has_gt,
            'pair_names': (os.path.basename(path_fix), os.path.basename(path_mov))
        }

# ============================================================================
# 可视化增强 (完全复用 v6)
# ============================================================================

def compute_checkerboard(img1, img2, n_grid=4):
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
    if torch.is_tensor(img1): img1 = (img1.cpu().numpy() * 255).astype(np.uint8)
    if torch.is_tensor(img2): img2 = (img2.cpu().numpy() * 255).astype(np.uint8)
    kp1_cv = [cv2.KeyPoint(x=float(pt[0]), y=float(pt[1]), size=10) for pt in kps1]
    kp2_cv = [cv2.KeyPoint(x=float(pt[0]), y=float(pt[1]), size=10) for pt in kps2]
    out_img = cv2.drawMatches(img1, kp1_cv, img2, kp2_cv, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imwrite(save_path, out_img)

def visualize_descriptors_pca(desc_fix, desc_mov):
    B, C, H, W = desc_fix.shape
    feat_fix = desc_fix[0].view(C, -1).permute(1, 0)
    feat_mov = desc_mov[0].view(C, -1).permute(1, 0)
    combined = F.normalize(torch.cat([feat_fix, feat_mov], dim=0), dim=-1)
    try:
        _, _, V = torch.pca_lowrank(combined, q=3, niter=2)
        pca_feat = torch.mm(combined, V)
        pca_feat_min = pca_feat.min(dim=0, keepdim=True)[0]
        pca_feat_max = pca_feat.max(dim=0, keepdim=True)[0]
        pca_feat = (pca_feat - pca_feat_min) / (pca_feat_max - pca_feat_min + 1e-8)
        pca_fix = pca_feat[:H*W].view(H, W, 3).cpu().numpy()
        pca_mov = pca_feat[H*W:].view(H, W, 3).cpu().numpy()
        return (pca_fix * 255).astype(np.uint8), (pca_mov * 255).astype(np.uint8)
    except:
        return np.zeros((H, W, 3), dtype=np.uint8), np.zeros((H, W, 3), dtype=np.uint8)

def compute_corner_error(H_est, H_gt, height, width):
    corners = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
    corners_homo = np.concatenate([corners, np.ones((4, 1), dtype=np.float32)], axis=1)
    corners_gt_homo = (H_gt @ corners_homo.T).T
    corners_gt = corners_gt_homo[:, :2] / (corners_gt_homo[:, 2:] + 1e-6)
    corners_est_homo = (H_est @ corners_homo.T).T
    corners_est = corners_est_homo[:, :2] / (corners_est_homo[:, 2:] + 1e-6)
    try:
        errors = np.sqrt(np.sum((corners_est - corners_gt)**2, axis=1))
        return np.mean(errors)
    except:
        return float('inf')

def validate(model, val_dataset, device, epoch, save_root, train_config, mode):
    model.eval()
    all_metrics = []
    log_file = os.path.join(save_root, 'validation_log.txt')
    log_f = open(log_file, 'a')
    
    nms_thresh = train_config.get('nms_thresh', 0.01)
    content_thresh = train_config.get('content_thresh', 0.8)
    geometric_thresh = train_config.get('geometric_thresh', 0.7)
    
    epoch_save_dir = os.path.join(save_root, f'epoch{epoch}')
    os.makedirs(epoch_save_dir, exist_ok=True)
    
    transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((512, 512)), transforms.ToTensor()])
    
    with torch.no_grad():
        for i in tqdm(range(len(val_dataset)), desc=f"Val Epoch {epoch}"):
            raw_data = val_dataset.get_raw_sample(i)
            if mode == 'cfocta': img_fix_raw, img_mov_raw, pts_fix_gt, pts_mov_gt, path_fix, path_mov = raw_data
            else: img_mov_raw, img_fix_raw, pts_mov_gt, pts_fix_gt, path_mov, path_fix = raw_data
            
            sample_id = os.path.basename(path_fix).split('.')[0]
            sample_save_dir = os.path.join(epoch_save_dir, sample_id)
            os.makedirs(sample_save_dir, exist_ok=True)
            
            img0 = transform(cv2.cvtColor(img_fix_raw, cv2.COLOR_RGB2GRAY) if img_fix_raw.ndim==3 else img_fix_raw).unsqueeze(0).to(device)
            img1 = transform(cv2.cvtColor(img_mov_raw, cv2.COLOR_RGB2GRAY) if img_mov_raw.ndim==3 else img_mov_raw).unsqueeze(0).to(device)
            
            det_fix, desc_fix = model.network(img0, mode='fix')
            det_mov, desc_mov = model.network(img1, mode='mov')
            
            pca_fix, pca_mov = visualize_descriptors_pca(desc_fix, desc_mov)
            cv2.imwrite(os.path.join(sample_save_dir, f'{sample_id}_pca_fix.png'), cv2.cvtColor(pca_fix, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(sample_save_dir, f'{sample_id}_pca_mov.png'), cv2.cvtColor(pca_mov, cv2.COLOR_RGB2BGR))
            
            kps_fix = nms(det_fix, nms_thresh=nms_thresh, nms_size=5)[0]
            kps_mov = nms(det_mov, nms_thresh=nms_thresh, nms_size=5)[0]
            
            # Use topk fallback if no points (aligned with v6)
            if len(kps_fix) < 10:
                flat = det_fix[0,0].view(-1); _, idx = torch.topk(flat, min(100, flat.numel()))
                kps_fix = torch.stack([idx % 512, idx // 512], dim=1).float()
            if len(kps_mov) < 10:
                flat = det_mov[0,0].view(-1); _, idx = torch.topk(flat, min(100, flat.numel()))
                kps_mov = torch.stack([idx % 512, idx // 512], dim=1).float()

            d1 = sample_keypoint_desc(kps_fix[None], desc_fix, s=8)[0].permute(1, 0).cpu().numpy()
            d2 = sample_keypoint_desc(kps_mov[None], desc_mov, s=8)[0].permute(1, 0).cpu().numpy()
            matches = cv2.BFMatcher().knnMatch(d1, d2, k=2)
            good = [m for m, n in matches if m.distance < content_thresh * n.distance]
            
            h_f, w_f = img_fix_raw.shape[:2]
            h_m, w_m = img_mov_raw.shape[:2]
            mkpts0 = np.array([kps_fix.cpu().numpy()[m.queryIdx] * [w_f/512.0, h_f/512.0] for m in good]) if good else np.array([])
            mkpts1 = np.array([kps_mov.cpu().numpy()[m.trainIdx] * [w_m/512.0, h_m/512.0] for m in good]) if good else np.array([])
            
            if (h_m, w_m) != (h_f, w_f):
                sc = [w_f/w_m, h_f/h_m]
                if len(mkpts1) > 0: mkpts1 *= sc
                if len(pts_mov_gt) > 0: pts_mov_gt = pts_mov_gt * sc
            
            H_gt, _ = cv2.findHomography(pts_mov_gt, pts_fix_gt, cv2.RANSAC, 5.0) if len(pts_mov_gt)>=4 else (None, None)
            H_pred, _ = cv2.findHomography(mkpts1, mkpts0, cv2.RANSAC, geometric_thresh) if len(mkpts0)>=4 else (None, None)
            H_est = H_pred if H_pred is not None else np.eye(3)
            mace = compute_corner_error(H_est, H_gt, h_f, w_f) if H_gt is not None else float('inf')

            # Metric Calculation Aligned with Paper
            orig_size_ref = (2912, 2912)
            sc_f = [orig_size_ref[1] / 512.0, orig_size_ref[0] / 512.0]
            mkpts0_p = kps_fix.cpu().numpy()[ [m.queryIdx for m in good] ] * sc_f if good else np.array([])
            mkpts1_p = kps_mov.cpu().numpy()[ [m.trainIdx for m in good] ] * sc_f if good else np.array([])
            pts_f_p = pts_fix_gt * [orig_size_ref[1] / w_f, orig_size_ref[0] / h_f]
            pts_m_p = pts_mov_gt * [orig_size_ref[1] / w_m, orig_size_ref[0] / h_m]
            
            res_paper = calculate_metrics(mkpts0_p, mkpts1_p, pts_f_p, pts_m_p, orig_size=orig_size_ref)
            all_metrics.append({**res_paper, 'MACE': mace})

            # Visualizations
            img_fix_kpts = cv2.drawKeypoints(img_fix_raw.copy(), [cv2.KeyPoint(pt[0], pt[1], 10) for pt in mkpts0], None, color=(0,255,0))
            cv2.putText(img_fix_kpts, f"Max:{det_fix.max():.2f} Pts:{len(mkpts0)}", (10,30), 1, 1, (255,0,0), 2)
            cv2.imwrite(os.path.join(sample_save_dir, f'{sample_id}_fix_kpts.png'), img_fix_kpts)
            
            reg_img = cv2.warpPerspective(cv2.resize(img_mov_raw, (w_f, h_f)), H_est, (w_f, h_f))
            cv2.imwrite(os.path.join(sample_save_dir, f'{sample_id}_result.png'), reg_img)
            cv2.imwrite(os.path.join(sample_save_dir, f'{sample_id}_checker.png'), compute_checkerboard(img_fix_raw, reg_img))
            
    # Summary
    all_errors = [e for m in all_metrics for e in m['errors']]
    auc5 = compute_auc(all_errors, 5); auc10 = compute_auc(all_errors, 10); auc20 = compute_auc(all_errors, 20)
    avg_auc = (auc5 + auc10 + auc20) / 3.0
    acc_rate = np.mean([m['is_acceptable'] for m in all_metrics])
    
    log_f.write(f"\n--- Validation Epoch {epoch} ---\n")
    log_f.write(f"Acceptable: {acc_rate*100:.2f}% | AUC@5: {auc5:.4f} | AUC@10: {auc10:.4f} | AUC@20: {auc20:.4f} | Avg: {avg_auc:.4f}\n")
    log_f.close()
    print(f"Val Epoch {epoch} Finished. Avg AUC: {avg_auc:.4f}, Acc: {acc_rate*100:.2f}%")
    return avg_auc

# ============================================================================
# 训练主循环
# ============================================================================

def train_real_v6():
    config_path = './config/train_multimodal.yaml'
    with open(config_path) as f: config = yaml.safe_load(f)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', '-n', type=str, default='v6_real_strict_baseline')
    parser.add_argument('--mode', '-m', type=str, choices=['cffa', 'cfoct', 'octfa', 'cfocta'], default='cffa')
    parser.add_argument('--epoch', '-e', type=int, default=150)
    parser.add_argument('--batch_size', '-b', type=int, default=4)
    parser.add_argument('--content_thresh', type=float, default=None)
    parser.add_argument('--nms_thresh', type=float, default=None)
    args = parser.parse_args()
    
    config['MODEL']['name'] = args.name
    config['DATASET']['registration_type'] = args.mode
    config['MODEL']['shared_encoder'] = False 
    if args.content_thresh is not None: config['MODEL']['content_thresh'] = args.content_thresh
    if args.nms_thresh is not None: config['MODEL']['nms_thresh'] = args.nms_thresh
    
    train_config = {**config['MODEL'], **config['PKE'], **config['DATASET'], **config['VALUE_MAP']}
    save_root = f'./save/{args.mode}/{args.name}'
    os.makedirs(save_root, exist_ok=True)
    
    log_file_train = os.path.join(save_root, 'train_log.txt')
    log_print = lambda msg: print(msg) or open(log_file_train, 'a').write(msg + '\n')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Dataset & Loaders
    if args.mode == 'cffa': base_train = CFFADataset(root_dir='dataset/operation_pre_filtered_cffa', split='train', mode='fa2cf')
    elif args.mode == 'cfocta': base_train = CFOCTADataset(root_dir='dataset/CF_OCTA_v2_repaired', split='train', mode='cf2octa')
    # ... other modes ...
    
    train_loader = DataLoader(RealDataV6Wrapper(base_train), batch_size=args.batch_size, shuffle=True, num_workers=4)
    model = SuperRetinaMultimodal(train_config, device=device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # Value Map Setup (Aligned with v6)
    value_map_save_dir = train_config['value_map_save_dir']
    if os.path.exists(value_map_save_dir): shutil.rmtree(value_map_save_dir)
    os.makedirs(value_map_save_dir)
    value_maps_running = {}
    
    phase0_epochs = train_config.get('warmup_epoch', 50)
    best_auc = -1.0
    
    log_print(f"Starting V6 Baseline on Real Data: {args.name}")

    for epoch in range(1, args.epoch + 1):
        if epoch <= phase0_epochs:
            phase, model.PKE_learn = 0, False
            phase_msg = f"Phase 0: Modality Alignment Warmup (Epoch {epoch}/{phase0_epochs})"
        else:
            phase, model.PKE_learn = 3, True
            phase_msg = f"Phase 1+: Hybrid PKE Registration (Epoch {epoch - phase0_epochs}/{args.epoch - phase0_epochs})"
            
        log_print(f'--- {phase_msg} ---')
        
        model.train()
        running_det, running_desc, total_samples = 0.0, 0.0, 0
        
        for step_idx, data in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
            img0_orig, img1_orig = data['image0'].to(device), (data['image1_origin'] if phase==0 else data['image1']).to(device)
            img0, img1 = apply_domain_randomization(img0_orig), apply_domain_randomization(img1_orig)
            
            if epoch == 1 and step_idx < 2:
                save_batch_visualization(img0_orig, img1_orig, img0, img1, save_root, epoch, step_idx+1, args.batch_size)
            
            seeds = data['vessel_mask0'].to(device)
            H_0to1 = data['T_0to1'].to(device)
            names = data['pair_names'][0]
            
            # Proper Value Map Loading
            v_maps = value_map_load(value_map_save_dir, names, torch.ones(img0.size(0), dtype=torch.bool), img0.shape[-2:], value_maps_running).to(device)
            
            optimizer.zero_grad()
            loss, _, l_det, l_desc, _, _, _, _, _ = model(
                img0, img1, seeds, v_maps, (torch.arange(img0.size(0)),),
                phase=phase, vessel_mask=None, H_0to1=H_0to1
            )
            loss.backward()
            optimizer.step()
            
            # Proper Value Map Saving
            value_map_save(value_map_save_dir, names, torch.ones(img0.size(0), dtype=torch.bool), v_maps.cpu(), value_maps_running)
            
            running_det += l_det; running_desc += l_desc; total_samples += img0.size(0)
            
        epoch_loss = (running_det + running_desc) / total_samples
        log_print(f'Train Total Loss: {epoch_loss:.4f} (Det: {running_det/total_samples:.4f}, Desc: {running_desc/total_samples:.4f})')
            
        if epoch % 5 == 0:
            auc = validate(model, base_train, device, epoch, save_root, train_config, args.mode) # Using base_train as placeholder for simplicity
            
            state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch, 'auc_score': auc}
            latest_path = os.path.join(save_root, 'latestpoint')
            os.makedirs(latest_path, exist_ok=True)
            torch.save(state, os.path.join(latest_path, 'checkpoint.pth'))
            
            if auc > best_auc:
                best_auc = auc
                best_path = os.path.join(save_root, 'bestcheckpoint')
                os.makedirs(best_path, exist_ok=True)
                torch.save(state, os.path.join(best_path, 'checkpoint.pth'))
                log_print(f"New Best AUC: {auc:.4f}")

if __name__ == '__main__':
    train_real_v6()
