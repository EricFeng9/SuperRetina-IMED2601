"""
基于原始 SuperRetina 训练流程，在真实多模态数据集上进行训练
用于与改进版 train_multimodal_v3_3.py 进行对比实验

主要修改：
1. 数据集从合成数据切换到真实多模态数据集（cffa/cfoct/octfa/cfocta）
2. 使用数据集的 get_raw_sample() 获取原始 fix 和 moving 图像对
3. 保持 SuperRetina 原始训练流程不变（PKE、Value Map、损失函数等）
4. 添加与 train_multimodal_v3_3.py 一致的验证和早停流程
"""

import torch
import os
import sys
import yaml
import cv2
import numpy as np
import argparse
import shutil
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# 添加本地模块路径
sys.path.append(os.getcwd())

from model.super_retina import SuperRetina
from common.common_util import nms, sample_keypoint_desc
from common.train_util import value_map_load, value_map_save
from dataset.CF_OCTA_v2_repaired.cf_octa_v2_repaired_dataset import CFOCTADataset
from dataset.operation_pre_filtered_cffa.operation_pre_filtered_cffa_dataset import CFFADataset
from dataset.operation_pre_filtered_cfoct.operation_pre_filtered_cfoct_dataset import CFOCTDataset
from dataset.operation_pre_filtered_octfa.operation_pre_filtered_octfa_dataset import OCTFADataset


class RealDatasetWrapper(Dataset):
    """
    包装真实数据集，使其兼容 SuperRetina 原始训练流程
    
    SuperRetina 原始训练流程需要的数据格式：
    - images: [B, 1, H, W] 单张图像
    - input_with_label: [B] bool tensor，标记哪些样本有标注（用于 PKE）
    - keypoint_positions: [B, 1, H, W] 关键点热图（全零，因为真实数据没有关键点标注）
    - label_names: [B] 图像名称列表
    
    本包装器从真实数据集读取 fix 和 moving 图像，返回配对数据用于训练
    """
    def __init__(self, base_dataset, img_size=512):
        """
        Args:
            base_dataset: 底层真实数据集 (CFOCTADataset/CFFADataset/CFOCTDataset/OCTFADataset)
            img_size: 目标图像尺寸
        """
        self.base_dataset = base_dataset
        self.img_size = img_size
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ])
    
    def __len__(self):
        # 返回 fix 和 moving 的总数（2倍）
        return len(self.base_dataset) * 2
    
    def __getitem__(self, idx):
        # 将索引映射到数据集样本和模式
        sample_idx = idx // 2
        is_moving = (idx % 2 == 1)
        
        # 获取原始数据: (img_fix, img_mov, pts_fix, pts_mov, path_fix, path_mov)
        raw_data = self.base_dataset.get_raw_sample(sample_idx)
        img_fix_raw, img_mov_raw, _, _, path_fix, path_mov = raw_data
        
        # 选择使用 fix 还是 moving
        if is_moving:
            img_raw = img_mov_raw
            path = path_mov
        else:
            img_raw = img_fix_raw
            path = path_fix
        
        # 确保图像为灰度
        if img_raw.ndim == 3:
            if img_raw.shape[2] == 3:
                img_gray = cv2.cvtColor(img_raw, cv2.COLOR_RGB2GRAY)
            else:
                img_gray = img_raw.squeeze()
        else:
            img_gray = img_raw
        
        # 转换为 Tensor [1, H, W]
        img_tensor = self.transform(img_gray)
        
        # 生成空的关键点标注 (SuperRetina 原始流程需要)
        keypoint_positions = torch.zeros(1, self.img_size, self.img_size, dtype=torch.float32)
        
        # 标记为有标注（启用 PKE 学习）
        input_with_label = torch.tensor(True, dtype=torch.bool)
        
        # 图像名称（用于 value map 管理）
        label_name = os.path.basename(path)
        
        return img_tensor, input_with_label, keypoint_positions, label_name


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


def validate(model, val_dataset, device, epoch, save_dir, log_file, train_config, mode):
    """
    验证函数：评估模型在真实数据集上的表现
    与 train_multimodal_v3_3.py 中的 validate 函数对齐
    """
    from measurement import calculate_metrics
    
    model.eval()
    all_metrics = []
    
    # 从配置中统一读取阈值
    nms_thresh = train_config.get('nms_thresh', 0.1)
    content_thresh = train_config.get('content_thresh', 0.7)
    geometric_thresh = train_config.get('geometric_thresh', 0.5)
    
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
            if mode == 'cfocta':
                img_fix_raw, img_mov_raw, pts_fix_gt, pts_mov_gt, path_fix, path_mov = raw_data
            elif mode == 'cffa':
                img_mov_raw, img_fix_raw, pts_mov_gt, pts_fix_gt, path_mov, path_fix = raw_data
            elif mode == 'cfoct':
                img_fix_raw, img_mov_raw, pts_fix_gt, pts_mov_gt, path_fix, path_mov = raw_data
            elif mode == 'octfa':
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
            
            # 提取特征
            det_fix, desc_fix = model.network(img0_tensor)
            det_mov, desc_mov = model.network(img1_tensor)
            
            # 有效区域屏蔽
            valid_mask = (img1_tensor > 0.05).float()
            valid_mask = -F.max_pool2d(-valid_mask, kernel_size=5, stride=1, padding=2)
            det_mov_masked = det_mov * valid_mask
            
            # 提取关键点
            kps_fix = nms(det_fix, nms_thresh=nms_thresh, nms_size=5)[0]
            kps_mov = nms(det_mov_masked, nms_thresh=nms_thresh, nms_size=5)[0]
            
            # 兜底策略
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
            
            # 统一尺寸处理（与 test_on_real.py 和 train_multimodal_v3_3.py 对齐）
            if (h_m, w_m) != (h_f, w_f):
                img_mov_resized = cv2.resize(img_mov_raw, (w_f, h_f), interpolation=cv2.INTER_LINEAR)
                scale_x = w_f / w_m
                scale_y = h_f / h_m
                
                if len(mkpts1) > 0:
                    mkpts1 = mkpts1 * [scale_x, scale_y]
                if len(kps_m_orig) > 0:
                    kps_m_orig = kps_m_orig * [scale_x, scale_y]
                if len(pts_mov_gt) > 0:
                    pts_mov_gt = pts_mov_gt * [scale_x, scale_y]
            else:
                img_mov_resized = img_mov_raw
            
            # 使用 measurement.py 进行评估
            metrics = calculate_metrics(
                img_origin=img_fix_raw, img_result=img_mov_resized,
                mkpts0=mkpts0, mkpts1=mkpts1,
                kpts0=kps_f_orig, kpts1=kps_m_orig,
                ctrl_pts0=pts_fix_gt, ctrl_pts1=pts_mov_gt
            )
            
            all_metrics.append(metrics)
            
            log_f.write(f"ID: {sample_id} | SR_ME: {metrics['SR_ME']} | SR_MAE: {metrics['SR_MAE']} | "
                       f"Rep: {metrics['Rep']:.4f} | MIR: {metrics['MIR']:.4f} | "
                       f"MeanErr: {metrics['mean_error']:.2f} px\n")
            
            # 保存可视化结果
            sample_save_dir = os.path.join(epoch_save_dir, sample_id)
            os.makedirs(sample_save_dir, exist_ok=True)
            
            cv2.imwrite(os.path.join(sample_save_dir, f'{sample_id}_fix.png'), img_fix_raw)
            cv2.imwrite(os.path.join(sample_save_dir, f'{sample_id}_moving.png'), img_mov_resized)

            if len(mkpts0) >= 4:
                H_pred, _ = cv2.findHomography(mkpts1, mkpts0, cv2.RANSAC, geometric_thresh)
                if H_pred is not None:
                    img_m_gray = cv2.cvtColor(img_mov_resized, cv2.COLOR_RGB2GRAY) if img_mov_resized.ndim == 3 else img_mov_resized
                    img_f_gray = cv2.cvtColor(img_fix_raw, cv2.COLOR_RGB2GRAY) if img_fix_raw.ndim == 3 else img_fix_raw
                    
                    reg_img = cv2.warpPerspective(img_m_gray, H_pred, (w_f, h_f))
                    cv2.imwrite(os.path.join(sample_save_dir, f'{sample_id}_moving_result.png'), reg_img)
                    
                    # 计算棋盘格可视化
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
                    
                    checker = compute_checkerboard(img_f_gray, reg_img, n_grid=4)
                    cv2.imwrite(os.path.join(sample_save_dir, f'{sample_id}_checkerboard.png'), checker)
            
            draw_matches(img_fix_raw, kps_f_orig, img_mov_resized, kps_m_orig, good_matches, 
                        os.path.join(sample_save_dir, f'{sample_id}_matches.png'))

    # 计算平均指标
    summary = {}
    for key in ['SR_ME', 'SR_MAE', 'Rep', 'MIR', 'mean_error', 'max_error']:
        vals = [m[key] for m in all_metrics if m[key] != float('inf')]
        summary[key] = np.mean(vals) if vals else 0.0
    
    log_f.write(f"\n--- Validation Summary ---\n")
    log_f.write(f"Overall SR_ME (Success Rate @5px):  {summary['SR_ME']*100:.2f}%\n")
    log_f.write(f"Overall SR_MAE (Success Rate @10px): {summary['SR_MAE']*100:.2f}%\n")
    log_f.write(f"Average Repeatability:              {summary['Rep']*100:.2f}%\n")
    log_f.write(f"Average Matching Inliers Ratio:     {summary['MIR']*100:.2f}%\n")
    log_f.write(f"Mean Registration Error:            {summary['mean_error']:.2f} px\n")
    log_f.write(f"Max Registration Error (Average):   {summary['max_error']:.2f} px\n")
    log_f.close()
    
    print(f'Validation Epoch {epoch} Finished. Mean Error: {summary["mean_error"]:.2f} px, SR_ME: {summary["SR_ME"]*100:.2f}%')
    return summary['mean_error']


def train_on_real():
    """
    在真实多模态数据集上使用原始 SuperRetina 流程进行训练
    """
    # 命令行参数
    parser = argparse.ArgumentParser(description="SuperRetina Training on Real Multimodal Data")
    parser.add_argument('--name', '-n', type=str, required=True, help='Experiment name')
    parser.add_argument('--mode', '-m', type=str, choices=['cffa', 'cfoct', 'octfa', 'cfocta'], 
                        required=True, help='Registration mode')
    parser.add_argument('--epoch', '-e', type=int, default=150, help='Number of training epochs')
    parser.add_argument('--batch_size', '-b', type=int, default=4, help='Batch size')
    parser.add_argument('--img_size', type=int, default=512, help='Image size for training')
    parser.add_argument('--pke_start_epoch', type=int, default=40, help='Epoch to start PKE learning')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    args = parser.parse_args()
    
    exp_name = args.name
    mode = args.mode
    
    # 加载配置文件
    config_path = './config/train.yaml'
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # 使用命令行参数覆盖配置
    config['MODEL']['num_epoch'] = args.epoch
    config['MODEL']['batch_size'] = args.batch_size
    config['PKE']['pke_start_epoch'] = args.pke_start_epoch
    
    # 修改模型保存路径
    save_root = f'./save_baseline/{mode}/{exp_name}'
    os.makedirs(save_root, exist_ok=True)
    
    # Value Map 保存目录
    value_map_dir = os.path.join(save_root, 'value_maps')
    config['VALUE_MAP']['value_map_save_dir'] = value_map_dir
    config['VALUE_MAP']['is_value_map_save'] = True
    
    train_config = {**config['MODEL'], **config['PKE'], **config['DATASET'], **config['VALUE_MAP']}
    
    # 设备配置
    device = torch.device(train_config['device'] if torch.cuda.is_available() else "cpu")
    
    # 日志文件
    log_file = os.path.join(save_root, 'validation_log.txt')
    train_log_file = os.path.join(save_root, 'train_log.txt')
    train_log = open(train_log_file, 'a', buffering=1)
    
    def log_print(msg):
        """同时输出到控制台和日志文件"""
        print(msg)
        train_log.write(msg + '\n')
        train_log.flush()
    
    log_print(f"Using device: {device} | Experiment: {exp_name} | Mode: {mode}")
    
    # 加载对应的真实数据集
    log_print(f"Loading {mode} dataset...")
    if mode == 'cfocta':
        train_base = CFOCTADataset(root_dir='dataset/CF_OCTA_v2_repaired', split='train', mode='cf2octa')
        val_base = CFOCTADataset(root_dir='dataset/CF_OCTA_v2_repaired', split='val', mode='cf2octa')
    elif mode == 'cffa':
        train_base = CFFADataset(root_dir='dataset/operation_pre_filtered_cffa', split='train', mode='fa2cf')
        val_base = CFFADataset(root_dir='dataset/operation_pre_filtered_cffa', split='val', mode='fa2cf')
    elif mode == 'cfoct':
        train_base = CFOCTDataset(root_dir='dataset/operation_pre_filtered_cfoct', split='train', mode='cf2oct')
        val_base = CFOCTDataset(root_dir='dataset/operation_pre_filtered_cfoct', split='val', mode='cf2oct')
    elif mode == 'octfa':
        train_base = OCTFADataset(root_dir='dataset/operation_pre_filtered_octfa', split='train', mode='fa2oct')
        val_base = OCTFADataset(root_dir='dataset/operation_pre_filtered_octfa', split='val', mode='fa2oct')
    
    # 包装数据集
    train_set = RealDatasetWrapper(train_base, img_size=args.img_size)
    
    log_print(f"Train set: {len(train_set)} images (from {len(train_base)} pairs)")
    log_print(f"Val set: {len(val_base)} pairs")
    
    # DataLoader
    batch_size = train_config['batch_size']
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    
    # 初始化原始 SuperRetina 模型
    log_print("Initializing SuperRetina model...")
    model = SuperRetina(train_config, device=device)
    
    # 加载预训练模型（如果指定）
    load_pre_trained_model = train_config.get('load_pre_trained_model', False)
    pretrained_path = train_config.get('pretrained_path', '')
    if load_pre_trained_model and os.path.exists(pretrained_path):
        log_print(f"Loading pretrained model from {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location=device)
        model.load_state_dict(checkpoint['net'])
    
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # 保存配置文件到实验目录
    config_save_path = os.path.join(save_root, 'train_config.yaml')
    with open(config_save_path, 'w') as f:
        yaml.dump({
            'command_args': vars(args),
            'train_config': train_config
        }, f)
    log_print(f"Config saved to {config_save_path}")
    
    # Value Map 初始化
    is_value_map_save = train_config['is_value_map_save']
    if is_value_map_save:
        if os.path.exists(value_map_dir):
            shutil.rmtree(value_map_dir)
        os.makedirs(value_map_dir)
    
    value_maps_running = {} if not is_value_map_save else None
    
    # 早停机制变量
    best_mean_error = float('inf')
    patience = 5
    patience_counter = 0
    best_val_error = float('inf')
    
    num_epochs = train_config['num_epoch']
    pke_start_epoch = train_config['pke_start_epoch']
    
    # 初始验证
    log_print("Running initial validation...")
    _ = validate(model, val_base, device, 0, save_root, log_file, train_config, mode)
    
    # 训练循环
    log_print("\n" + "="*50)
    log_print("Starting training with original SuperRetina pipeline...")
    log_print("="*50 + "\n")
    
    for epoch in range(1, num_epochs + 1):
        # PKE 学习开关
        model.PKE_learn = (epoch >= pke_start_epoch)
        phase_name = f"PKE {'ON' if model.PKE_learn else 'OFF'}"
        
        log_print(f'Epoch {epoch}/{num_epochs} | {phase_name}')
        model.train()
        
        running_loss_det = 0.0
        running_loss_desc = 0.0
        num_learned_pts = 0
        num_input_with_label = 0
        num_input_descriptor = 0
        
        for images, input_with_label, keypoint_positions, label_names in tqdm(train_loader, desc=f"Train Epoch {epoch}"):
            batch_size_cur = images.shape[0]
            learn_index = torch.where(input_with_label)
            
            images = images.to(device)
            keypoint_positions = keypoint_positions.to(device)
            
            # 加载 Value Maps
            value_maps = value_map_load(value_map_dir, label_names, input_with_label,
                                       images.shape[-2:], value_maps_running)
            value_maps = value_maps.to(device)
            
            optimizer.zero_grad()
            
            with torch.set_grad_enabled(True):
                loss, number_pts_one, print_loss_detector_one, print_loss_descriptor_one, \
                    enhanced_label_pts, enhanced_label, detector_pred, loss_detector_num, loss_descriptor_num \
                    = model(images, keypoint_positions, value_maps, learn_index)
                
                loss.backward()
                optimizer.step()
            
            # 更新 Value Maps
            if len(learn_index[0]) != 0:
                value_maps = value_maps.cpu()
                value_map_save(value_map_dir, label_names, input_with_label, value_maps, value_maps_running)
            
            running_loss_det += print_loss_detector_one * len(learn_index[0])
            running_loss_desc += print_loss_descriptor_one * batch_size_cur
            num_input_with_label += loss_detector_num
            num_learned_pts += number_pts_one
            num_input_descriptor += loss_descriptor_num
        
        # 计算平均损失
        avg_det_loss = running_loss_det / max(num_input_with_label, 1)
        avg_desc_loss = running_loss_desc / max(num_input_descriptor, 1)
        avg_learned_pts = num_learned_pts / max(num_input_with_label, 1)
        
        log_print(f'Train Total Loss: {avg_det_loss + avg_desc_loss:.4f} '
                 f'(Det: {avg_det_loss:.4f}, Desc: {avg_desc_loss:.4f}, '
                 f'AvgPts: {avg_learned_pts:.1f})')
        
        # 每 5 个 Epoch 进行一次验证
        if epoch % 5 == 0:
            mean_error = validate(model, val_base, device, epoch, save_root, log_file, train_config, mode)
            
            state = {
                'net': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'mean_error': mean_error
            }
            
            # 保存最新模型
            latest_dir = os.path.join(save_root, 'latestpoint')
            os.makedirs(latest_dir, exist_ok=True)
            torch.save(state, os.path.join(latest_dir, 'checkpoint.pth'))
            with open(os.path.join(latest_dir, 'checkpoint_info.txt'), 'w') as f:
                f.write(f'Latest Checkpoint\nEpoch: {epoch}\nMean Error: {mean_error:.4f} px\n')
            
            # 保存最佳模型
            if mean_error < best_mean_error:
                log_print(f"New Best Mean Error: {mean_error:.4f} px (Previous: {best_mean_error:.4f} px)")
                best_mean_error = mean_error
                best_dir = os.path.join(save_root, 'bestcheckpoint')
                os.makedirs(best_dir, exist_ok=True)
                torch.save(state, os.path.join(best_dir, 'checkpoint.pth'))
                with open(os.path.join(best_dir, 'checkpoint_info.txt'), 'w') as f:
                    f.write(f'Best Checkpoint\nEpoch: {epoch}\nMean Error: {mean_error:.4f} px\n')
            
            # 早停机制（epoch >= 100 后启用）
            if epoch >= 100:
                if mean_error < best_val_error:
                    best_val_error = mean_error
                    patience_counter = 0
                    log_print(f'[Early Stopping] Validation Mean Error improved to {best_val_error:.4f} px. Reset patience counter.')
                else:
                    patience_counter += 1
                    log_print(f'[Early Stopping] Validation Mean Error did not improve. Patience: {patience_counter}/{patience}')
                
                if patience_counter >= patience:
                    log_print(f'Early stopping triggered at epoch {epoch}. Best validation Mean Error: {best_val_error:.4f} px')
                    break
    
    train_log.close()
    log_print("\nTraining completed!")
    log_print(f"Best model saved to {os.path.join(save_root, 'bestcheckpoint/checkpoint.pth')}")


if __name__ == '__main__':
    train_on_real()
