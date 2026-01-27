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
from common.train_util import value_map_load, value_map_save, affine_images
from common.common_util import nms, sample_keypoint_desc
from gen_data_enhance_v2 import apply_domain_randomization, save_batch_visualization


# ============================================================================
# 数据集包装器 (对齐 v6 接口)
# ============================================================================

class RealDataV6Wrapper(Dataset):
    """
    为了对齐 v6 策略，模拟产生 image1_origin (几何对齐版本)
    """
    def __init__(self, base_dataset, img_size=512):
        self.base_dataset = base_dataset
        self.img_size = img_size
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        raw_data = self.base_dataset.get_raw_sample(idx)
        img_fix_raw, img_mov_raw, pts_fix, pts_mov, path_fix, path_mov = raw_data
        
        def to_gray(img):
            if img.ndim == 3:
                return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.shape[2] == 3 else img.squeeze()
            return img

        # 1. 转换基础图像
        img0_512 = cv2.resize(to_gray(img_fix_raw), (self.img_size, self.img_size))
        img1_512 = cv2.resize(to_gray(img_mov_raw), (self.img_size, self.img_size))
        
        h_f, w_f = img_fix_raw.shape[:2]
        h_m, w_m = img_mov_raw.shape[:2]
        pts_fix_512 = pts_fix * [self.img_size / w_f, self.img_size / h_f]
        pts_mov_512 = pts_mov * [self.img_size / w_m, self.img_size / h_m]
        
        # 2. 计算 GT Homography (从 Fix 到 Moving)
        H_0to1 = np.eye(3, dtype=np.float32)
        H_inv = np.eye(3, dtype=np.float32)
        if len(pts_fix_512) >= 4:
            H, _ = cv2.findHomography(pts_fix_512, pts_mov_512, cv2.RANSAC, 5.0)
            if H is not None:
                H_0to1 = H.astype(np.float32)
                try: H_inv = np.linalg.inv(H_0to1)
                except: pass

        # 3. 构造 Phase 0 所需的 "几何已对齐" 图像 (image1_origin)
        # 将 Moving 图像 Warp 到 Fix 空间
        img1_aligned = cv2.warpPerspective(img1_512, H_inv, (self.img_size, self.img_size))
        
        # 4. 构造 PKE 初始点图 (种子点)
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
            'vessel_mask0': torch.from_numpy(seed_map).unsqueeze(0), # 对照组: 种子点作为掩码引导
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

def validate(model, val_dataset, device, epoch, save_root, train_config, mode):
    from measurement_SuperRetina import calculate_metrics, compute_auc
    model.eval()
    all_metrics = []
    log_file = os.path.join(save_root, 'validation_log.txt')
    log_f = open(log_file, 'a')
    
    nms_thresh = train_config.get('nms_thresh', 0.01)
    content_thresh = train_config.get('content_thresh', 0.7)
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
            cv2.imwrite(os.path.join(sample_save_dir, f'{sample_id}_desc_pca_fix.png'), cv2.cvtColor(pca_fix, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(sample_save_dir, f'{sample_id}_desc_pca_mov.png'), cv2.cvtColor(pca_mov, cv2.COLOR_RGB2BGR))
            
            kps_fix = nms(det_fix, nms_thresh=nms_thresh, nms_size=5)[0]
            kps_mov = nms(det_mov, nms_thresh=nms_thresh, nms_size=5)[0]
            
            if len(kps_fix) < 4:
                flat = det_fix[0,0].view(-1); _, idx = torch.topk(flat, 100)
                kps_fix = torch.stack([idx % 512, idx // 512], dim=1).float()
            if len(kps_mov) < 4:
                flat = det_mov[0,0].view(-1); _, idx = torch.topk(flat, 100)
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
            
            H_pred, _ = cv2.findHomography(mkpts1, mkpts0, cv2.RANSAC, geometric_thresh) if len(mkpts0) >= 4 else (None, None)
            H_est = H_pred if H_pred is not None else np.eye(3)
            
            # 保存关键点与配准预览
            img_fix_kpts = cv2.drawKeypoints(img_fix_raw.copy(), [cv2.KeyPoint(pt[0], pt[1], 10) for pt in mkpts0], None, color=(0,255,0))
            cv2.imwrite(os.path.join(sample_save_dir, f'{sample_id}_fix_kpts.png'), img_fix_kpts)
            reg_img = cv2.warpPerspective(img_mov_raw, H_est, (w_f, h_f))
            cv2.imwrite(os.path.join(sample_save_dir, f'{sample_id}_moving_result.png'), reg_img)
            cv2.imwrite(os.path.join(sample_save_dir, f'{sample_id}_checkerboard.png'), compute_checkerboard(img_fix_raw, reg_img))
            
            res = calculate_metrics(mkpts0, mkpts1, pts_fix_gt, pts_mov_gt, orig_size=(w_f, h_f))
            all_metrics.append({'errors': res['errors'], 'Acc': res['is_acceptable']})
            
    errors = [e for m in all_metrics for e in m['errors']]
    avg_auc = (compute_auc(errors, 5) + compute_auc(errors, 10) + compute_auc(errors, 20)) / 3.0 if errors else 0.0
    acc = np.mean([m['Acc'] for m in all_metrics]) if all_metrics else 0.0
    log_f.write(f"Epoch {epoch} | Avg AUC@5-20: {avg_auc:.4f} | Acc: {acc:.4f}\n")
    log_f.close()
    return avg_auc

# ============================================================================
# 训练主循环 (严格对齐 v6)
# ============================================================================

def train_real_v6():
    config_path = './config/train_multimodal.yaml'
    with open(config_path) as f: config = yaml.safe_load(f)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', '-n', type=str, default='v6_real_strict_control')
    parser.add_argument('--mode', '-m', type=str, choices=['cffa', 'cfoct', 'octfa', 'cfocta'], default='cffa')
    parser.add_argument('--epoch', '-e', type=int, default=150)
    parser.add_argument('--batch_size', '-b', type=int, default=4)
    parser.add_argument('--content_thresh', type=float, default=None)
    parser.add_argument('--nms_thresh', type=float, default=None)
    args = parser.parse_args()
    
    config['MODEL']['name'] = args.name
    config['DATASET']['registration_type'] = args.mode
    config['MODEL']['shared_encoder'] = False # Dual-Path

    # Override config values if provided via command line
    if args.content_thresh is not None: config['MODEL']['content_thresh'] = args.content_thresh
    if args.nms_thresh is not None: config['MODEL']['nms_thresh'] = args.nms_thresh
    
    train_config = {**config['MODEL'], **config['PKE'], **config['DATASET'], **config['VALUE_MAP']}
    save_root = f'./save/{args.mode}/{args.name}'
    os.makedirs(save_root, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_print = lambda msg: print(msg) or open(os.path.join(save_root, 'train_log.txt'), 'a').write(msg + '\n')
    
    # 1. 数据集加载
    if args.mode == 'cffa':
        base_train = CFFADataset(root_dir='dataset/operation_pre_filtered_cffa', split='train', mode='fa2cf')
        base_val = CFFADataset(root_dir='dataset/operation_pre_filtered_cffa', split='val', mode='fa2cf')
    # ...
    
    train_loader = DataLoader(RealDataV6Wrapper(base_train), batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    model = SuperRetinaMultimodal(train_config, device=device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    phase0_epochs = 30
    best_auc = -1.0
    
    log_print(f"Starting V6 Strategy on Real Data: {args.name}")

    for epoch in range(1, args.epoch + 1):
        if epoch <= phase0_epochs:
            phase, model.PKE_learn, phase_name = 0, False, f"Phase 0: Modality Alignment Warmup"
        else:
            phase, model.PKE_learn, phase_name = 3, True, f"Phase 1+: Hybrid PKE Registration"
            
        log_print(f'--- {phase_name} (Epoch {epoch}/{args.epoch}) ---')
        model.train()
        
        for step_idx, data in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
            img0_orig = data['image0'].to(device)
            img1_orig = data['image1_origin'].to(device) if phase == 0 else data['image1'].to(device)
            
            # 域随机化 (对齐 v6)
            img0, img1 = apply_domain_randomization(img0_orig), apply_domain_randomization(img1_orig)
            
            # 可视化前两个 Batch (对齐 v6)
            if epoch == 1 and step_idx < 2:
                save_batch_visualization(img0_orig, img1_orig, img0, img1, save_root, epoch, step_idx+1, args.batch_size)
            
            seeds = data['vessel_mask0'].to(device) # 真值控制点作为种子点
            H_0to1 = data['T_0to1'].to(device)
            
            optimizer.zero_grad()
            # 调用 forward, 不适用血管分割图则传 None
            loss, _, l_det, l_desc, _, _, _, _, _ = model(
                img0, img1, seeds, 
                value_map=torch.zeros_like(seeds),
                learn_index=(torch.arange(img0.size(0)),),
                phase=phase, 
                vessel_mask=None, # <<< 对照组关掉血管掩码约束
                H_0to1=H_0to1
            )
            loss.backward()
            optimizer.step()
            
        if epoch % 5 == 0:
            auc = validate(model, base_val, device, epoch, save_root, train_config, args.mode)
            if auc > best_auc:
                best_auc = auc
                torch.save({'net': model.state_dict(), 'auc': auc}, os.path.join(save_root, 'bestcheckpoint/checkpoint.pth'))
                log_print(f"New Best AUC: {auc:.4f}")

if __name__ == '__main__':
    train_real_v6()
