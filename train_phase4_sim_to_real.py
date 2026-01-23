import torch
import os
import sys
import yaml
import cv2
import numpy as np
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import argparse

# 添加本地模块路径
sys.path.append(os.getcwd())

# 导入真实数据集
from dataset.operation_pre_filtered_octfa.operation_pre_filtered_octfa_dataset import OCTFADataset
from dataset.operation_pre_filtered_cfoct.operation_pre_filtered_cfoct_dataset import CFOCTDataset
from dataset.operation_pre_filtered_cffa.operation_pre_filtered_cffa_dataset import CFFADataset
from dataset.CF_OCTA_v2_repaired.cf_octa_v2_repaired_dataset import CFOCTADataset

from model.super_retina_multimodal import SuperRetinaMultimodal
from common.train_util import value_map_load, value_map_save
from common.common_util import nms, sample_keypoint_desc


class RealDatasetWrapper(torch.utils.data.Dataset):
    """
    真实数据集包装器 - 将真实数据集的输出格式统一成训练所需的格式
    真实数据有 GT 关键点和 H_computed，可以用于完整的 PKE 训练
    """
    def __init__(self, real_dataset):
        self.dataset = real_dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        fix, moving_orig, moving_gt, fix_path, moving_path, T_0to1 = self.dataset[idx]
        
        # 获取原始关键点 (用于 Anchor Loss)
        raw_data = self.dataset.get_raw_sample(idx)
        # raw_data: (img_fix, img_mov, pts_fix, pts_mov, path_fix, path_mov)
        pts_fix = raw_data[2]  # numpy array [N, 2]
        pts_mov = raw_data[3]  # numpy array [N, 2]
        
        # 转换为灰度图 (取RGB的第一个通道)
        if fix.shape[0] == 3:
            fix = fix[0:1]  # [1, H, W]
        if moving_orig.shape[0] == 3:
            moving_orig = moving_orig[0:1]
        if moving_gt.shape[0] == 3:
            moving_gt = moving_gt[0:1]
        
        # 归一化到 [0, 1]（真实数据集返回的是 [-1, 1]）
        moving_orig = (moving_orig + 1) / 2
        moving_gt = (moving_gt + 1) / 2
        
        # 将关键点转换为热力图格式 (512x512 -> 64x64)
        H, W = 512, 512
        Hc, Wc = H // 8, W // 8
        
        # 创建关键点热力图
        keypoints_heatmap = np.zeros((Hc, Wc), dtype=np.float32)
        
        # 将关键点缩放到 64x64 尺度
        pts_fix_scaled = pts_fix / 8.0  # 512 -> 64
        
        for pt in pts_fix_scaled:
            x, y = int(pt[0]), int(pt[1])
            if 0 <= x < Wc and 0 <= y < Hc:
                keypoints_heatmap[y, x] = 1.0
        
        # 转换为 tensor
        keypoints_tensor = torch.from_numpy(keypoints_heatmap).float()
        
        return {
            'image0': fix,
            'image1': moving_gt,
            'image1_origin': moving_orig,
            'T_0to1': T_0to1,
            'gt_keypoints_fix': torch.from_numpy(pts_fix).float(),  # [N, 2] 原始尺度
            'gt_keypoints_mov': torch.from_numpy(pts_mov).float(),  # [N, 2] 原始尺度
            'keypoints_heatmap': keypoints_tensor,  # [64, 64] 用于可视化
            'pair_names': os.path.basename(fix_path),  # 修改为单个字符串，让 DataLoader 处理成 [str, str...]
            'is_real': True  # 标记为真实数据
        }


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
    content_thresh = train_config.get('content_thresh', 0.7)
    geometric_thresh = train_config.get('geometric_thresh', 0.7)
    
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
        print("初始化验证集: 使用固定种子选取10个样本...")
        g = torch.Generator()
        g.manual_seed(2024)
        
        dataset_len = len(val_loader.dataset)
        indices = torch.randperm(dataset_len, generator=g).tolist()
        
        selected_indices = set(indices[:10] if dataset_len >= 10 else indices)
        
        for batch_idx, data in enumerate(tqdm(val_loader, desc="缓存验证数据")):
            if batch_idx in selected_indices:
                val_cache.append(data)
                if len(val_cache) >= len(selected_indices):
                    break
    
    with torch.no_grad():
        for data in tqdm(val_cache, desc=f"验证 Epoch {epoch}"):
            img0 = data['image0'].to(device)
            img1 = data['image1'].to(device)
            img1_origin = data['image1_origin'].to(device)
            
            pair_names = data.get('pair_names', 'sample_unknown')
            if isinstance(pair_names, (list, tuple)):
                sample_name = str(pair_names[0])
            else:
                sample_name = str(pair_names)
            
            sample_id = os.path.splitext(os.path.basename(sample_name))[0]
            
            det_fix, desc_fix = model.network(img0)
            det_mov, desc_mov = model.network(img1)
            
            for b in range(img0.shape[0]):
                valid_mask = (img1[b:b+1] > 0.05).float()
                import torch.nn.functional as F
                valid_mask = -F.max_pool2d(-valid_mask, kernel_size=5, stride=1, padding=2)
                det_mov_masked = det_mov[b:b+1] * valid_mask
                
                kps_fix = nms(det_fix[b:b+1], nms_thresh=nms_thresh, nms_size=5)[0]
                kps_mov = nms(det_mov_masked, nms_thresh=nms_thresh, nms_size=5)[0]
                
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
                
                mse = 10000.0
                img_reg = None
                good = []

                if len(kps_fix) >= 4 and len(kps_mov) >= 4:
                    desc_fix_samp = sample_keypoint_desc(kps_fix[None], desc_fix[b:b+1], s=8)[0]
                    desc_mov_samp = sample_keypoint_desc(kps_mov[None], desc_mov[b:b+1], s=8)[0]
                    
                    d1 = desc_fix_samp.permute(1, 0).cpu().numpy()
                    d2 = desc_mov_samp.permute(1, 0).cpu().numpy()
                    
                    bf = cv2.BFMatcher()
                    matches = bf.knnMatch(d1, d2, k=2)
                    
                    for m, n in matches:
                        if m.distance < content_thresh * n.distance:
                            good.append(m)
                
                if len(good) >= 4:
                    src_pts = np.float32([kps_fix[m.queryIdx].cpu().numpy() for m in good]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kps_mov[m.trainIdx].cpu().numpy() for m in good]).reshape(-1, 1, 2)
                    
                    M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, geometric_thresh)
                    
                    if M is not None:
                        img_warped_np = img1[b, 0].cpu().numpy() * 255.0
                        h, w = img_warped_np.shape
                        img_reg = cv2.warpPerspective(img_warped_np, M, (w, h))
                        
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

                cv2.imwrite(os.path.join(sample_save_dir, f'{sample_id}_fix_raw.png'), fix_np)
                cv2.imwrite(os.path.join(sample_save_dir, f'{sample_id}_moving_origin_raw.png'), mov_in_np)
                cv2.imwrite(os.path.join(sample_save_dir, f'{sample_id}_moving_warped_raw.png'), mov_aff_np)

                cv2.imwrite(os.path.join(sample_save_dir, f'{sample_id}_fixed_kps.png'), fix_with_kps)
                cv2.imwrite(os.path.join(sample_save_dir, f'{sample_id}_affined_moving_kps.png'), mov_aff_with_kps)

                reg_np = img_reg.astype(np.uint8) if img_reg is not None else fix_np
                cv2.imwrite(os.path.join(sample_save_dir, f'{sample_id}_registered.png'), reg_np)

                # 棋盘格拼接图
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
                
                draw_matches(fix_np, kps_fix.cpu().numpy(), mov_aff_np, kps_mov.cpu().numpy(), good, 
                             os.path.join(sample_save_dir, f'{sample_id}_matches.png'))

    avg_mse = np.mean(mse_list) if len(mse_list) > 0 else float('inf')
    log_f.write(f'AVERAGE MSE: {avg_mse:.4f}\n')
    log_f.close()
    
    print(f'验证 Epoch {epoch} 完成. 平均 MSE: {avg_mse:.4f}')
    return avg_mse, val_cache


def train_phase4():
    """
    阶段 4：全量真实数据 PKE 微调
    直接使用真实数据进行完整的 Phase 3 PKE 训练
    利用真实数据中的 GT 关键点和计算出的 H_computed 进行监督
    """
    config_path = './config/train_multimodal.yaml'
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件未找到: {config_path}")
        
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # 命令行参数解析
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', '-n', type=str, help='实验名称', default=None)
    parser.add_argument('--srcName', type=str, help='源实验名称(加载预训练权重)', required=True)
    parser.add_argument('--mode', '-m', type=str, choices=['cffa', 'cfoct', 'octfa', 'cfocta'], 
                        help='配准模式', default=None)
    parser.add_argument('--epoch', '-e', type=int, help='训练轮数', default=100)
    parser.add_argument('--batch_size', '-b', type=int, help='批量大小', default=4)
    parser.add_argument('--geometric_thresh', '-g', type=float, help='RANSAC几何阈值', default=0.7)
    parser.add_argument('--content_thresh', '-c', type=float, help='Lowe比率阈值', default=0.8)
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
    
    exp_name = train_config.get('name', 'phase4_exp')
    reg_type = train_config['registration_type']
    save_root = f'./save/{reg_type}/{exp_name}'
    os.makedirs(save_root, exist_ok=True)
    
    log_file = os.path.join(save_root, 'validation_log.txt')
    train_log_file = os.path.join(save_root, 'train_log.txt')
    
    device = torch.device(train_config['device'] if torch.cuda.is_available() else "cpu")
    
    train_log = open(train_log_file, 'a', buffering=1)
    
    def log_print(msg):
        """同时输出到控制台和日志文件"""
        print(msg)
        train_log.write(msg + '\n')
        train_log.flush()
    
    log_print(f"使用设备: {device} | 实验名称: {exp_name}")
    log_print(f"阶段 4: 全量真实数据 PKE 微调")

    # ===== 数据加载 =====
    batch_size = train_config['batch_size']
    
    # 加载真实数据集
    log_print(f"加载真实数据集 (模式: {reg_type})...")
    if reg_type == 'cffa':
        real_train_raw = CFFADataset(
            root_dir='/data/student/Fengjunming/SuperRetina/dataset/operation_pre_filtered_cffa',
            split='train',
            mode='cf2fa'
        )
        real_val_raw = CFFADataset(
            root_dir='/data/student/Fengjunming/SuperRetina/dataset/operation_pre_filtered_cffa',
            split='val',
            mode='cf2fa'
        )
    elif reg_type == 'cfoct':
        real_train_raw = CFOCTDataset(
            root_dir='/data/student/Fengjunming/SuperRetina/dataset/operation_pre_filtered_cfoct',
            split='train',
            mode='cf2oct'
        )
        real_val_raw = CFOCTDataset(
            root_dir='/data/student/Fengjunming/SuperRetina/dataset/operation_pre_filtered_cfoct',
            split='val',
            mode='cf2oct'
        )
    elif reg_type == 'octfa':
        real_train_raw = OCTFADataset(
            root_dir='/data/student/Fengjunming/SuperRetina/dataset/operation_pre_filtered_octfa',
            split='train',
            mode='fa2oct'
        )
        real_val_raw = OCTFADataset(
            root_dir='/data/student/Fengjunming/SuperRetina/dataset/operation_pre_filtered_octfa',
            split='val',
            mode='fa2oct'
        )
    elif reg_type == 'cfocta':
        real_train_raw = CFOCTADataset(
            root_dir='/data/student/Fengjunming/SuperRetina/dataset/CF_OCTA_v2_repaired',
            split='train',
            mode='cf2octa'
        )
        real_val_raw = CFOCTADataset(
            root_dir='/data/student/Fengjunming/SuperRetina/dataset/CF_OCTA_v2_repaired',
            split='val',
            mode='cf2octa'
        )
    else:
        raise ValueError(f"不支持的配准模式: {reg_type}")
    
    # 包装真实数据集
    train_set = RealDatasetWrapper(real_train_raw)
    val_set = RealDatasetWrapper(real_val_raw)
    
    n_train = len(train_set)
    log_print(f"真实训练样本数: {n_train}")
    
    # 创建 DataLoader
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=False)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=4)
    
    # ===== 初始化模型 =====
    model = SuperRetinaMultimodal(train_config, device=device)
    
    # 加载预训练权重
    pretrained_path = f'./save/{reg_type}/{args.srcName}/bestcheckpoint/checkpoint.pth'
    if not os.path.exists(pretrained_path):
        raise FileNotFoundError(f"预训练权重未找到: {pretrained_path}")
    
    log_print(f"从以下路径加载预训练权重: {pretrained_path}")
    checkpoint = torch.load(pretrained_path, map_location=device)
    model.load_state_dict(checkpoint['net'])
    
    # 使用较小的学习率 (1/10)
    lr = 1e-5
    optimizer = optim.Adam(model.parameters(), lr=lr)
    log_print(f"学习率设置为: {lr}")
    
    num_epochs = train_config['num_epoch']
    
    # Value Map 配置
    is_value_map_save = train_config['is_value_map_save']
    value_map_save_dir = train_config['value_map_save_dir']
    
    if is_value_map_save and not os.path.exists(value_map_save_dir):
        os.makedirs(value_map_save_dir)
        
    value_maps_running = {} if not is_value_map_save else None
    best_mse = float('inf')
    
    # 早停机制
    patience = 5
    patience_counter = 0
    best_val_mse = float('inf')

    # 初始验证
    val_cache = []
    log_print("运行初始验证...")
    _, val_cache = validate(model, val_loader, device, 0, save_root, log_file, train_config, val_cache=val_cache)

    # ===== 训练循环 =====
    for epoch in range(1, num_epochs + 1):
        log_print(f'Epoch {epoch}/{num_epochs} | Phase 4: 全量真实数据 PKE 微调')
        model.train()
        
        # Phase 4: PKE 完全开启
        model.PKE_learn = True
            
        running_loss_det = 0.0
        running_loss_desc = 0.0
        total_samples = 0
        
        pbar = tqdm(train_loader, desc=f"训练 Epoch {epoch}")
        for batch_data in pbar:
            img0 = batch_data['image0'].to(device)
            img1 = batch_data['image1'].to(device)
            H_0to1 = batch_data['T_0to1'].to(device)
            keypoints_heatmap = batch_data['keypoints_heatmap'].unsqueeze(1).to(device)  # [B, 1, 64, 64]
            
            batch_size = img0.size(0)
            
            # 创建 learn_index
            learn_index = (torch.arange(batch_size, device=device),)
            input_with_labels = torch.ones(batch_size, dtype=torch.bool).to(device)
            
            # 处理 pair_names
            pair_names_raw = batch_data['pair_names']
            if isinstance(pair_names_raw, (list, tuple)):
                names = [str(n) for n in pair_names_raw]
            else:
                names = [str(pair_names_raw)]
            
            # 加载 value_maps
            value_maps = value_map_load(value_map_save_dir, names, input_with_labels, 
                                      img0.shape[-2:], value_maps_running)
            value_maps = value_maps.to(device)
            
            optimizer.zero_grad()
            
            with torch.set_grad_enabled(True):
                # 使用 Phase 3 完整 PKE 训练（真实数据不使用 vessel_mask）
                loss, number_pts, loss_det_item, loss_desc_item, enhanced_kp, enhanced_label, det_pred, n_det, n_desc = \
                    model(img0, img1, keypoints_heatmap, value_maps, learn_index,
                          phase=3, vessel_mask=None, H_0to1=H_0to1)
                    
                loss.backward()
                optimizer.step()
            
            # 保存 value_maps
            if len(learn_index[0]) > 0:
                value_maps = value_maps.cpu()
                value_map_save(value_map_save_dir, names, input_with_labels, value_maps, value_maps_running)
                    
            running_loss_det += loss_det_item
            running_loss_desc += loss_desc_item
            total_samples += batch_size
            
            # 更新进度条
            pbar.set_postfix({
                'Det': f'{running_loss_det/max(total_samples,1):.4f}',
                'Desc': f'{running_loss_desc/max(total_samples,1):.4f}'
            })

        epoch_loss = (running_loss_det + running_loss_desc) / total_samples
        log_print(f'训练总损失: {epoch_loss:.4f}')
        log_print(f'  Det Loss: {running_loss_det/max(total_samples,1):.4f}, Desc Loss: {running_loss_desc/max(total_samples,1):.4f}')
        log_print(f'样本总数: {total_samples}')
        
        # 每 5 个 Epoch 进行一次验证
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
            with open(os.path.join(latest_dir, 'checkpoint_info.txt'), 'w') as f:
                f.write(f'Latest Checkpoint\nEpoch: {epoch}\nMSE: {avg_mse:.4f}\n')
            
            # 保存最佳模型
            if avg_mse < best_mse:
                log_print(f"新的最佳 MSE: {avg_mse:.4f} (之前: {best_mse:.4f})")
                best_mse = avg_mse
                best_dir = os.path.join(save_root, 'bestcheckpoint')
                os.makedirs(best_dir, exist_ok=True)
                torch.save(state, os.path.join(best_dir, 'checkpoint.pth'))
                with open(os.path.join(best_dir, 'checkpoint_info.txt'), 'w') as f:
                    f.write(f'Best Checkpoint\nEpoch: {epoch}\nMSE: {avg_mse:.4f}\n')
            
            # 早停机制
            if avg_mse < best_val_mse:
                best_val_mse = avg_mse
                patience_counter = 0
                log_print(f'[早停] 验证 MSE 改善至 {best_val_mse:.4f}. 重置耐心计数器.')
            else:
                patience_counter += 1
                log_print(f'[早停] 验证 MSE 未改善. 耐心: {patience_counter}/{patience}')
            
            if patience_counter >= patience:
                log_print(f'早停触发于 epoch {epoch}. 最佳验证 MSE: {best_val_mse:.4f}')
                break
    
    train_log.close()
    log_print("训练完成!")

if __name__ == '__main__':
    train_phase4()
