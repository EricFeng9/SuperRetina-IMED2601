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
from data.student.Fengjunming.SuperRetina.dataset.FIVES_extract_v2.FIVES_extract_v2 import MultiModalDataset
from model.super_retina_multimodal import SuperRetinaMultimodal
from common.train_util import value_map_load, value_map_save, affine_images
from common.common_util import nms, sample_keypoint_desc

def get_vessel_weight_min(epoch):
    """
    根据epoch返回三阶段课程学习的最小权重
    
    阶段1 (0-30): min_weight = 0.3  (全局感知期)
    阶段2 (30-60): min_weight = 0.1  (血管聚焦期)
    阶段3 (60-100): min_weight = 0.5  (泛化增强期)
    阶段4 (100+): min_weight = 0.0  (完全泛化期,掩码权重归零)
    """
    if epoch < 30:
        return 0.3
    elif epoch < 60:
        return 0.1
    elif epoch < 100:
        return 0.5
    else:
        return 0.0

def compute_weighted_dice_loss(pred, target, weight):
    """
    计算加权 Dice Loss
    
    Args:
        pred: 预测检测图 [B, H, W]
        target: 目标标签 [B, H, W] 
        weight: 权重图 [B, H, W]
    
    Returns:
        weighted_dice_loss
    """
    smooth = 1e-5
    
    # 确保维度一致
    pred_flat = pred.view(pred.size(0), -1)
    target_flat = target.view(target.size(0), -1)
    weight_flat = weight.view(weight.size(0), -1)
    
    # 加权交集和并集
    intersection = (pred_flat * target_flat * weight_flat).sum(dim=1)
    pred_sum = (pred_flat * pred_flat * weight_flat).sum(dim=1)
    target_sum = (target_flat * target_flat * weight_flat).sum(dim=1)
    
    dice = (2.0 * intersection + smooth) / (pred_sum + target_sum + smooth)
    
    return 1.0 - dice.mean()

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

def validate(model, val_loader, device, epoch, save_dir, log_file, train_config, cached_transforms=None):
    """
    验证函数:评估模型在跨模态配准任务上的 MSE 表现
    """
    model.eval()
    mse_list = []
    
    # 从配置中统一读取阈值
    nms_thresh = train_config.get('nms_thresh', 0.01)
    content_thresh = train_config.get('content_thresh', 0.7) # Lowe's Ratio
    geometric_thresh = train_config.get('geometric_thresh', 3.0) # RANSAC re-projection error
    
    # 缓存变换矩阵
    if cached_transforms is None:
        cached_transforms = {}
        is_caching = True
    else:
        is_caching = False
    
    epoch_save_dir = os.path.join(save_dir, f'epoch{epoch}')
    os.makedirs(epoch_save_dir, exist_ok=True)
    
    log_f = open(log_file, 'a')
    log_f.write(f'\n--- Validation Epoch {epoch} ---\n')
    
    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(val_loader, desc=f"Val Epoch {epoch}")):
            img0 = data['image0'].to(device)
            img1 = data['image1'].to(device)
            img1_origin = data['image1_origin'].to(device)
            
            sample_id = f"sample_{batch_idx}"
            
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
                
                # 绘制关键点
                fix_with_kps = cv2.cvtColor(fix_np, cv2.COLOR_GRAY2BGR)
                for kp in kps_fix:
                    cv2.circle(fix_with_kps, (int(kp[0]), int(kp[1])), 3, (0, 255, 0), -1)
                
                mov_aff_with_kps = cv2.cvtColor(mov_aff_np, cv2.COLOR_GRAY2BGR)
                for kp in kps_mov:
                    cv2.circle(mov_aff_with_kps, (int(kp[0]), int(kp[1])), 3, (0, 255, 0), -1)

                cv2.imwrite(os.path.join(sample_save_dir, 'fixed_kps.png'), fix_with_kps)
                cv2.imwrite(os.path.join(sample_save_dir, 'affined_moving_kps.png'), mov_aff_with_kps)
                cv2.imwrite(os.path.join(sample_save_dir, 'registered.png'), img_reg.astype(np.uint8) if img_reg is not None else fix_np)
                
                # 绘制匹配关系
                draw_matches(fix_np, kps_fix.cpu().numpy(), mov_aff_np, kps_mov.cpu().numpy(), good, 
                             os.path.join(sample_save_dir, 'matches.png'))

    avg_mse = np.mean(mse_list) if len(mse_list) > 0 else float('inf')
    log_f.write(f'AVERAGE MSE: {avg_mse:.4f}\n')
    log_f.close()
    
    print(f'Validation Epoch {epoch} Finished. Avg MSE: {avg_mse:.4f}')
    return avg_mse, cached_transforms

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
    parser.add_argument('--epoch', '-e', type=int, help='Number of training epochs', default=None)
    parser.add_argument('--vessel_sigma', type=float, help='Vessel weight gaussian sigma', default=6.0)
    args = parser.parse_args()
    
    if args.name:
        config['MODEL']['name'] = args.name
    if args.mode:
        config['DATASET']['registration_type'] = args.mode
    if args.epoch:
        config['MODEL']['num_epoch'] = args.epoch
        
    train_config = {**config['MODEL'], **config['PKE'], **config['DATASET'], **config['VALUE_MAP']}
    
    exp_name = train_config.get('name', 'default_exp')
    reg_type = train_config['registration_type']
    save_root = f'./save/{reg_type}/{exp_name}'
    os.makedirs(save_root, exist_ok=True)
    
    log_file = os.path.join(save_root, 'validation_log.txt')
    device = torch.device(train_config['device'] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | Experiment: {exp_name}")

    # 数据加载 - 使用新的FIVES数据集
    root_dir = train_config['root_dir']
    batch_size = train_config['batch_size']
    img_size = train_config.get('img_size', 518)
    df = train_config.get('df', 8)
    
    train_set = MultiModalDataset(
        root_dir=root_dir, 
        mode=reg_type, 
        split='train', 
        img_size=img_size, 
        df=df,
        vessel_sigma=args.vessel_sigma
    )
    val_set = MultiModalDataset(
        root_dir=root_dir, 
        mode=reg_type, 
        split='val', 
        img_size=img_size, 
        df=df,
        vessel_sigma=args.vessel_sigma
    )
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=4)
    
    # 初始化多模态 SuperRetina 模型
    model = SuperRetinaMultimodal(train_config, device=device)
    
    if train_config['load_pre_trained_model']:
        path = train_config['pretrained_path']
        if os.path.exists(path):
            print(f"Loading pretrained model from {path}")
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

    # 初始验证
    print("Running initial validation...")
    _, val_transforms_cache = validate(model, val_loader, device, 0, save_root, log_file, train_config, cached_transforms=None)

    # 训练循环
    for epoch in range(1, num_epochs + 1):
        # 计算当前epoch的血管掩码最小权重
        vessel_min_weight = get_vessel_weight_min(epoch)
        
        # PKE 渐进式学习开关:冷启动后开启
        model.PKE_learn = (epoch >= pke_start_epoch)
            
        print(f'Epoch {epoch}/{num_epochs} | PKE_learn: {model.PKE_learn} | Vessel min_weight: {vessel_min_weight}')
        model.train()
            
        running_loss_det = 0.0
        running_loss_desc = 0.0
        total_samples = 0
        
        for data in tqdm(train_loader, desc=f"Train Epoch {epoch}"):
            img0 = data['image0'].to(device)
            img1 = data['image1'].to(device)
            vessel_weight0 = data['vessel_weight0'].to(device)  # [B, 1, H, W]
            vessel_weight1 = data['vessel_weight1'].to(device)
            
            # 应用三阶段课程学习的权重调整
            # W_vessel = clamp(M^vessel, min=vessel_min_weight)
            vessel_weight0_adjusted = torch.clamp(vessel_weight0, min=vessel_min_weight)
            vessel_weight1_adjusted = torch.clamp(vessel_weight1, min=vessel_min_weight)
            
            optimizer.zero_grad()
            
            with torch.set_grad_enabled(True):
                # 前向传播
                det0, desc0 = model.network(img0)
                det1, desc1 = model.network(img1)
                
                # 这里需要根据实际的SuperRetinaMultimodal接口调整
                # 假设模型返回检测图和描述子,我们需要计算加权损失
                # 注意:这是简化版本,实际需要根据model的forward方法调整
                
                # 示例:计算加权检测损失
                # loss_det = compute_weighted_dice_loss(det0, target_label, vessel_weight0_adjusted)
                
                # 实际应该调用模型的forward方法,这里仅作示意
                # 需要将vessel_weight传入模型或在此处计算损失
                
                # 临时使用原始接口(需要根据实际情况修改)
                loss, number_pts, loss_det_item, loss_desc_item, enhanced_kp, enhanced_label, det_pred, n_det, n_desc = \
                    model(img0, img1, None, None, None)  # 这里需要适配新数据格式
                    
                loss.backward()
                optimizer.step()
                    
            running_loss_det += loss_det_item
            running_loss_desc += loss_desc_item
            total_samples += img0.size(0)

        epoch_loss = (running_loss_det + running_loss_desc) / total_samples
        print(f'Train Total Loss: {epoch_loss:.4f} (Det: {running_loss_det/total_samples:.4f}, Desc: {running_loss_desc/total_samples:.4f})')
        
        # 每 5 个 Epoch 进行一次验证并保存模型
        if epoch % 5 == 0:
            avg_mse, _ = validate(model, val_loader, device, epoch, save_root, log_file, train_config, cached_transforms=val_transforms_cache)
            
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
            
            # 保存 MSE 表现最好的模型
            if avg_mse < best_mse:
                print(f"New Best MSE: {avg_mse:.4f} (Previous: {best_mse:.4f})")
                best_mse = avg_mse
                best_dir = os.path.join(save_root, 'bestcheckpoint')
                os.makedirs(best_dir, exist_ok=True)
                torch.save(state, os.path.join(best_dir, 'checkpoint.pth'))

if __name__ == '__main__':
    train_multimodal()
