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
from tqdm import tqdm
import argparse
import random

# 添加本地模块路径
sys.path.append(os.getcwd())

# 使用新数据集脚本
from dataset.FIVES_extract_v2.FIVES_extract_v2 import MultiModalDataset
from model.super_retina_multimodal import SuperRetinaMultimodal
from common.train_util import value_map_load, value_map_save, affine_images
from common.common_util import nms, sample_keypoint_desc
from common.vessel_keypoint_extractor import extract_vessel_keypoints, extract_vessel_keypoints_fallback



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

def validate(model, val_loader, device, epoch, save_dir, log_file, train_config, val_cache=None):
    """
    验证函数:评估模型在跨模态配准任务上的 MSE 表现
    """
    model.eval()
    mse_list = []
    
    # 从配置中统一读取阈值
    nms_thresh = train_config.get('nms_thresh', 0.01)
    content_thresh = train_config.get('content_thresh', 0.7) # Lowe's Ratio
    geometric_thresh = train_config.get('geometric_thresh', 0.7) # RANSAC re-projection error
    
    # Initialize cache if needed
    if val_cache is None:
        val_cache = []
        is_caching = True
    else:
        is_caching = False
    
    epoch_save_dir = os.path.join(save_dir, f'epoch{epoch}')
    os.makedirs(epoch_save_dir, exist_ok=True)
    
    log_f = open(log_file, 'a')
    log_f.write(f'\n--- Validation Epoch {epoch} ---\n')
    
    # Fill cache if empty (First run logic)
    if len(val_cache) == 0:
        print("Initializing Validation Set with Fixed Seed and 10 Samples...")
        g = torch.Generator()
        g.manual_seed(2024) # Fixed seed for reproducibility
        
        dataset_len = len(val_loader.dataset)
        indices = torch.randperm(dataset_len, generator=g).tolist()
        
        # Take 10 samples or all if less than 10
        selected_indices = set(indices[:10] if dataset_len >= 10 else indices)
        
        # Fetch specific samples
        for batch_idx, data in enumerate(tqdm(val_loader, desc="Caching Val Data")):
            if batch_idx in selected_indices:
                val_cache.append(data)
                if len(val_cache) >= len(selected_indices):
                    break
    
    with torch.no_grad():
        for data in tqdm(val_cache, desc=f"Val Epoch {epoch}"):
            img0 = data['image0'].to(device)
            img1 = data['image1'].to(device)
            img1_origin = data['image1_origin'].to(device)
            
            # Extract sample name
            pair_names = data.get('pair_names', [['sample_unknown']])[0]
            if isinstance(pair_names, (list, tuple)):
                sample_name = pair_names[0]
            else:
                sample_name = pair_names
            
            sample_id = os.path.splitext(os.path.basename(str(sample_name)))[0]
            
            # 提取跨模态特征
            det_fix, desc_fix = model.network(img0)
            det_mov, desc_mov = model.network(img1)
            
            for b in range(img0.shape[0]):
                # 有效区域屏蔽,防止边缘伪影干扰关键点提取
                valid_mask = (img1[b:b+1] > 0.05).float()
                import torch.nn.functional as F
                valid_mask = -F.max_pool2d(-valid_mask, kernel_size=5, stride=1, padding=2)
                det_mov_masked = det_mov[b:b+1] * valid_mask
                
                # 提取关键点
                kps_fix = nms(det_fix[b:b+1], nms_thresh=nms_thresh, nms_size=5)[0]
                kps_mov = nms(det_mov_masked, nms_thresh=nms_thresh, nms_size=5)[0]
                
                # 兜底策略:如果关键点太少,强制取响应最高的前100个点
                if len(kps_fix) < 10:
                     flat_det = det_fix[b, 0].view(-1)
                     _, idx = torch.topk(flat_det, min(100, flat_det.numel()))
                     y = idx // det_fix.shape[3]
                     x = idx % det_fix.shape[3]
                     kps_fix = torch.stack([x, y], dim=1).float()

                if len(kps_mov) < 10:
                     flat_det = det_mov_masked[0, 0].view(-1)
                     if flat_det.max() > 0:
                        _, idx = torch.topk(flat_det, min(100, flat_det.numel()))
                        y = idx // det_mov.shape[3]
                        x = idx % det_mov.shape[3]
                        kps_mov = torch.stack([x, y], dim=1).float()
                
                mse = 10000.0 # 默认较大值表示配准失败
                img_reg = None
                good = []

                if len(kps_fix) >= 4 and len(kps_mov) >= 4:
                    # 采样描述子并进行特征匹配
                    desc_fix_samp = sample_keypoint_desc(kps_fix[None], desc_fix[b:b+1], s=8)[0]
                    desc_mov_samp = sample_keypoint_desc(kps_mov[None], desc_mov[b:b+1], s=8)[0]
                    
                    d1 = desc_fix_samp.permute(1, 0).cpu().numpy()
                    d2 = desc_mov_samp.permute(1, 0).cpu().numpy()
                    
                    bf = cv2.BFMatcher()
                    matches = bf.knnMatch(d1, d2, k=2)
                    
                    # 使用配置中的 content_thresh 作为 Ratio Test 阈值
                    for m, n in matches:
                        if m.distance < content_thresh * n.distance:
                            good.append(m)
                
                if len(good) >= 4:
                    src_pts = np.float32([kps_fix[m.queryIdx].cpu().numpy() for m in good]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kps_mov[m.trainIdx].cpu().numpy() for m in good]).reshape(-1, 1, 2)
                    
                    # 利用配置中的 geometric_thresh 作为 RANSAC 容忍误差
                    M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, geometric_thresh)
                    
                    if M is not None:
                        img_warped_np = img1[b, 0].cpu().numpy() * 255.0
                        h, w = img_warped_np.shape
                        img_reg = cv2.warpPerspective(img_warped_np, M, (w, h))
                        
                        # 计算与 Ground Truth (原始对齐图像) 的均方误差
                        img_gt_np = img1_origin[b, 0].cpu().numpy() * 255.0
                        mse = np.mean((img_reg - img_gt_np) ** 2)
                
                mse_list.append(mse)
                
                # 保存可视化结果
                sample_save_dir = os.path.join(epoch_save_dir, sample_id)
                os.makedirs(sample_save_dir, exist_ok=True)
                
                fix_np = (img0[b, 0].cpu().numpy() * 255).astype(np.uint8)
                mov_in_np = (img1_origin[b, 0].cpu().numpy() * 255).astype(np.uint8)
                mov_aff_np = (img1[b, 0].cpu().numpy() * 255).astype(np.uint8)
                
                # ===== 新增：可视化从血管分割图中提取的关键点 =====
                if 'vessel_mask0' in data:
                    vessel_mask_np = (data['vessel_mask0'][b, 0].cpu().numpy() * 255).astype(np.uint8)
                    
                    # 提取关键点
                    try:
                        vessel_keypoints_np = extract_vessel_keypoints(vessel_mask_np, min_distance=8)
                    except:
                        try:
                            vessel_keypoints_np = extract_vessel_keypoints_fallback(vessel_mask_np, min_distance=8)
                        except:
                            vessel_keypoints_np = (vessel_mask_np > 127).astype(np.float32)
                    
                    # 在固定图上绘制提取的关键点 (GT on fix)
                    vessel_kps_vis = cv2.cvtColor(fix_np, cv2.COLOR_GRAY2BGR)
                    keypoint_coords = np.column_stack(np.where(vessel_keypoints_np > 0.5))
                    for coord in keypoint_coords:
                        cv2.circle(vessel_kps_vis, (int(coord[1]), int(coord[0])), 4, (0, 0, 255), -1)  # 红色点
                    cv2.imwrite(os.path.join(sample_save_dir, f'{sample_id}_vessel_keypoints_extracted.png'), vessel_kps_vis)

                    # ===== 新增：在已仿射的 moving 图像上绘制 GT 关键点 =====
                    if 'T_0to1' in data and keypoint_coords.size > 0:
                        H = data['T_0to1'][b].cpu().numpy()  # 3x3
                        pts = np.stack([
                            keypoint_coords[:, 1].astype(np.float32),  # x
                            keypoint_coords[:, 0].astype(np.float32),  # y
                            np.ones(len(keypoint_coords), dtype=np.float32)
                        ], axis=0)  # [3, N]
                        pts_mov = H @ pts  # [3, N]
                        xs_mov = pts_mov[0] / (pts_mov[2] + 1e-6)
                        ys_mov = pts_mov[1] / (pts_mov[2] + 1e-6)

                        h_img, w_img = mov_aff_np.shape
                        valid = (xs_mov >= 0) & (xs_mov < w_img) & (ys_mov >= 0) & (ys_mov < h_img)
                        xs_mov = xs_mov[valid]
                        ys_mov = ys_mov[valid]

                        mov_gt_kps_vis = cv2.cvtColor(mov_aff_np, cv2.COLOR_GRAY2BGR)
                        for x_m, y_m in zip(xs_mov, ys_mov):
                            cv2.circle(mov_gt_kps_vis, (int(x_m), int(y_m)), 4, (0, 0, 255), -1)  # 红色 GT 点
                        cv2.imwrite(os.path.join(sample_save_dir, f'{sample_id}_affined_moving_gt_kps.png'), mov_gt_kps_vis)
                
                # 绘制关键点
                fix_with_kps = cv2.cvtColor(fix_np, cv2.COLOR_GRAY2BGR)
                for kp in kps_fix:
                    cv2.circle(fix_with_kps, (int(kp[0]), int(kp[1])), 3, (0, 255, 0), -1)
                
                mov_aff_with_kps = cv2.cvtColor(mov_aff_np, cv2.COLOR_GRAY2BGR)
                for kp in kps_mov:
                    cv2.circle(mov_aff_with_kps, (int(kp[0]), int(kp[1])), 3, (0, 255, 0), -1)

                # 原始图像可视化：fix / moving_origin / moving_warped
                cv2.imwrite(os.path.join(sample_save_dir, f'{sample_id}_fix_raw.png'), fix_np)
                cv2.imwrite(os.path.join(sample_save_dir, f'{sample_id}_moving_origin_raw.png'), mov_in_np)
                cv2.imwrite(os.path.join(sample_save_dir, f'{sample_id}_moving_warped_raw.png'), mov_aff_np)

                cv2.imwrite(os.path.join(sample_save_dir, f'{sample_id}_fixed_kps.png'), fix_with_kps)
                cv2.imwrite(os.path.join(sample_save_dir, f'{sample_id}_affined_moving_kps.png'), mov_aff_with_kps)

                reg_np = img_reg.astype(np.uint8) if img_reg is not None else fix_np
                cv2.imwrite(os.path.join(sample_save_dir, f'{sample_id}_registered.png'), reg_np)

                # ===== 新增：registered 与 fix 的 4x4 棋盘格拼接图 =====
                h_img, w_img = fix_np.shape
                tile_h = h_img // 4
                tile_w = w_img // 4
                checker = np.zeros_like(fix_np)
                for i in range(4):
                    for j in range(4):
                        y0 = i * tile_h
                        y1 = h_img if i == 3 else (i + 1) * tile_h
                        x0 = j * tile_w
                        x1 = w_img if j == 3 else (j + 1) * tile_w
                        if (i + j) % 2 == 0:
                            checker[y0:y1, x0:x1] = fix_np[y0:y1, x0:x1]
                        else:
                            checker[y0:y1, x0:x1] = reg_np[y0:y1, x0:x1]
                cv2.imwrite(os.path.join(sample_save_dir, f'{sample_id}_fix_registered_checkerboard.png'), checker)
                
                # 绘制匹配关系
                draw_matches(fix_np, kps_fix.cpu().numpy(), mov_aff_np, kps_mov.cpu().numpy(), good, 
                             os.path.join(sample_save_dir, f'{sample_id}_matches.png'))

    avg_mse = np.mean(mse_list) if len(mse_list) > 0 else float('inf')
    log_f.write(f'AVERAGE MSE: {avg_mse:.4f}\n')
    log_f.close()
    
    print(f'Validation Epoch {epoch} Finished. Avg MSE: {avg_mse:.4f}')
    return avg_mse, val_cache

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

    # 数据加载 - 使用新的FIVES数据集
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
    val_set = MultiModalDataset(
        root_dir=root_dir, 
        mode=reg_type, 
        split='val', 
        img_size=img_size, 
        df=df
    )
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=4)
    
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
    best_mse = float('inf')
    
    # 早停机制变量 (仅在epoch >= 100后启用)
    patience = 5  # 验证损失连续5次不下降则早停
    patience_counter = 0
    best_val_mse = float('inf')

    # 初始验证
    val_cache = []
    log_print("Running initial validation...")
    _, val_cache = validate(model, val_loader, device, 0, save_root, log_file, train_config, val_cache=val_cache)

    pke_start_epoch = train_config.get('pke_start_epoch', 40) # 默认40以后开启PKE
    
    # 训练循环
    for epoch in range(1, num_epochs + 1):
        # Determine Training Phase
        if epoch <= 20:
            phase = 1 # Warmup
            model.PKE_learn = False
            phase_name = "Phase 1: Warmup (Desc + GT Det)"
        elif epoch <= 40:
            phase = 2 # Geo Consistency
            model.PKE_learn = False
            phase_name = "Phase 2: Geo Consistency (Desc + GT Det + Geo Det)"
        else:
            phase = 3 # Joint PKE
            model.PKE_learn = True
            phase_name = "Phase 3: Joint PKE"
            
        log_print(f'Epoch {epoch}/{num_epochs} | {phase_name}')
        model.train()
            
        running_loss_det = 0.0
        running_loss_desc = 0.0
        total_samples = 0
        
        for data in tqdm(train_loader, desc=f"Train Epoch {epoch}"):
            img0 = data['image0'].to(device)
            img1 = data['image1'].to(device)
            # ===== 关键修改：从完整血管分割图中提取稀疏的分叉点 =====
            # 这些分叉点将作为训练时的监督信号，引导模型学习独特的关键点
            vessel_mask_full = data['vessel_mask0']  # [B, 1, H, W]
            vessel_keypoints_batch = []
            
            for b in range(vessel_mask_full.shape[0]):
                # 转换为 numpy 格式 (H, W)
                mask_np = (vessel_mask_full[b, 0].cpu().numpy() * 255).astype(np.uint8)
                
                # 提取关键点
                try:
                    keypoints = extract_vessel_keypoints(mask_np, min_distance=8)
                except:
                    try:
                        keypoints = extract_vessel_keypoints_fallback(mask_np, min_distance=8)
                    except:
                        # 如果提取失败，使用原始掩码（但这会导致之前的问题）
                        keypoints = (mask_np > 127).astype(np.float32)
                        print("点位提取失败")
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
                # 调用模型 forward 方法，传入关键点掩码（而不是完整血管掩码）作为初始标签
                # 同时传入完整血管掩码 vessel_mask_full 用于 PKE 候选点过滤
                loss, number_pts, loss_det_item, loss_desc_item, enhanced_kp, enhanced_label, det_pred, n_det, n_desc = \
                    model(img0, img1, vessel_keypoints, value_maps, learn_index,
                          phase=phase, vessel_mask=vessel_mask_full, H_0to1=H_0to1)
                    
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
            avg_mse, _ = validate(model, val_loader, device, epoch, save_root, log_file, train_config, val_cache=val_cache)
            
            state = {
                'net': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'mse': avg_mse
            }
            
            # 保存最新模型
            latest_dir = os.path.join(save_root, 'latestpoint')
            os.makedirs(latest_dir, exist_ok=True)
            torch.save(state, os.path.join(latest_dir, 'checkpoint.pth'))
            # 保存epoch信息
            with open(os.path.join(latest_dir, 'checkpoint_info.txt'), 'w') as f:
                f.write(f'Latest Checkpoint\nEpoch: {epoch}\nMSE: {avg_mse:.4f}\n')
            
            # 保存 MSE 表现最好的模型
            if avg_mse < best_mse:
                log_print(f"New Best MSE: {avg_mse:.4f} (Previous: {best_mse:.4f})")
                best_mse = avg_mse
                best_dir = os.path.join(save_root, 'bestcheckpoint')
                os.makedirs(best_dir, exist_ok=True)
                torch.save(state, os.path.join(best_dir, 'checkpoint.pth'))
                # 保存epoch信息
                with open(os.path.join(best_dir, 'checkpoint_info.txt'), 'w') as f:
                    f.write(f'Best Checkpoint\nEpoch: {epoch}\nMSE: {avg_mse:.4f}\n')
            
            # 早停机制 (仅在 epoch >= 100 后启用)
            if epoch >= 100:
                if avg_mse < best_val_mse:
                    best_val_mse = avg_mse
                    patience_counter = 0
                    log_print(f'[Early Stopping] Validation MSE improved to {best_val_mse:.4f}. Reset patience counter.')
                else:
                    patience_counter += 1
                    log_print(f'[Early Stopping] Validation MSE did not improve. Patience: {patience_counter}/{patience}')
                
                if patience_counter >= patience:
                    log_print(f'Early stopping triggered at epoch {epoch}. Best validation MSE: {best_val_mse:.4f}')
                    break
    
    # 训练结束，关闭日志文件
    train_log.close()

if __name__ == '__main__':
    train_multimodal()
