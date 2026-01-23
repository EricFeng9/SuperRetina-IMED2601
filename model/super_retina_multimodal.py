import random
import sys
import time

from model.pke_module import pke_learn

from torch.nn import functional as F
import torch
import torch.nn as nn

from loss.dice_loss import DiceBCELoss, DiceLoss
from loss.triplet_loss import triplet_margin_loss_gor, triplet_margin_loss_gor_one, sos_reg

from common.common_util import remove_borders, sample_keypoint_desc, simple_nms, nms, \
    sample_descriptors
from common.train_util import get_gaussian_kernel, affine_images


def double_conv(in_channels, out_channels):
    """
    双层卷积模块，用于上采样过程中的特征融合
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )


class SuperRetinaMultimodal(nn.Module):
    """
    SuperRetina 多模态配准网络
    基于 Siamese 架构，利用共享权重的编码器处理不同模态（CF, FA, OCT）的图像
    """
    def __init__(self, config=None, device='cpu', n_class=1):
        super().__init__()

        self.PKE_learn = True
        self.relu = torch.nn.ReLU(inplace=True)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5, d1, d2 = 64, 64, 128, 128, 256, 256, 256
        
        # --- 共享权重的编码器 (Shared Encoder) ---
        # 按照训练计划，使用共享权重的编码器来提取不同模态的共性结构特征
        self.conv1a = torch.nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
        self.conv1b = torch.nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
        self.conv2a = torch.nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.conv2b = torch.nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
        self.conv3a = torch.nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.conv3b = torch.nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
        self.conv4a = torch.nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        self.conv4b = torch.nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)

        # --- 描述子头部 (Descriptor Head) ---
        # 生成 256 维的特征描述子，用于跨模态匹配
        self.convDa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convDb = torch.nn.Conv2d(c5, d1, kernel_size=4, stride=2, padding=0)
        self.convDc = torch.nn.Conv2d(d1, d2, kernel_size=1, stride=1, padding=0)
        self.trans_conv = nn.ConvTranspose2d(d1, d2, 2, stride=2)

        # --- 检测器头部 (Detector Head) ---
        # 基于 U-Net 结构的上采样路径，生成关键点检测概率图
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dconv_up3 = double_conv(c3 + c4, c3)
        self.dconv_up2 = double_conv(c2 + c3, c2)
        self.dconv_up1 = double_conv(c1 + c2, c1)
        self.conv_last = nn.Conv2d(c1, n_class, kernel_size=1)

        if config is not None:
            self.config = config
            self.nms_size = config.get('nms_size', 10)
            self.nms_thresh = config.get('nms_thresh', 0.01)
            self.scale = 8
            self.dice = DiceLoss()
            # 高斯核用于生成软标签 (Gaussian Smoothing for soft labels)
            # 提供默认值以防配置文件中缺少这些键
            kernel_size = config.get('gaussian_kernel_size', 21)
            sigma = config.get('gaussian_sigma', 3)
            self.kernel = get_gaussian_kernel(kernlen=kernel_size, nsig=sigma).to(device)

        self.to(device)

    def network(self, x):
        """
        前向传播基础网络：提取检测图 P 和描述子张量 D
        """
        # 编码阶段
        x = self.relu(self.conv1a(x))
        conv1 = self.relu(self.conv1b(x))
        x = self.pool(conv1)
        x = self.relu(self.conv2a(x))
        conv2 = self.relu(self.conv2b(x))
        x = self.pool(conv2)
        x = self.relu(self.conv3a(x))
        conv3 = self.relu(self.conv3b(x))
        x = self.pool(conv3)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))

        # --- 描述子分支 ---
        cDa = self.relu(self.convDa(x))
        cDb = self.relu(self.convDb(cDa))
        desc = self.convDc(cDb)
        # L2 归一化，保证描述子在欧氏空间的可比性
        dn = torch.norm(desc, p=2, dim=1)
        desc = desc.div(torch.unsqueeze(dn, 1))
        # 上采样回原始分辨率的 1/8 (或根据需要调整)
        desc = self.trans_conv(desc)

        # --- 检测器分支 (U-Net style) ---
        cPa = self.upsample(x)
        cPa = torch.cat([cPa, conv3], dim=1)
        cPa = self.dconv_up3(cPa)
        
        cPa = self.upsample(cPa)
        cPa = torch.cat([cPa, conv2], dim=1)
        cPa = self.dconv_up2(cPa)
        
        cPa = self.upsample(cPa)
        cPa = torch.cat([cPa, conv1], dim=1)
        cPa = self.dconv_up1(cPa)

        semi = self.conv_last(cPa)
        semi = torch.sigmoid(semi) # 输出关键点概率图 [0, 1]

        return semi, desc

    def descriptor_loss(self, detector_pred, label_point_positions, descriptor_pred,
                        affine_descriptor_pred, grid_inverse, affine_detector_pred=None):
        """
        计算描述子损失 (Triplet Loss)
        强制相同解剖点在不同模态下的特征描述子尽可能接近
        """
        if not self.PKE_learn:
            detector_pred[:] = 0  # 冷启动阶段仅使用初始种子点
        # 将初始标签点权重大幅提升，确保它们被选中进行匹配
        detector_pred[label_point_positions == 1] = 10
        
        # 采样关键点及其对应的描述子
        # descriptors: 固定图像上的特征
        # affine_descriptors: 运动图像（变换后）上对应的特征
        descriptors, affine_descriptors, keypoints = \
            sample_descriptors(detector_pred, descriptor_pred, affine_descriptor_pred, grid_inverse,
                               nms_size=self.nms_size, nms_thresh=self.nms_thresh, scale=self.scale,
                               affine_detector_pred=affine_detector_pred)

        positive = []
        negatives_hard = []
        negatives_random = []
        anchor = []
        D = descriptor_pred.shape[1]
        
        for i in range(len(affine_descriptors)):
            if affine_descriptors[i].shape[1] == 0:
                continue
            descriptor = descriptors[i]
            affine_descriptor = affine_descriptors[i]

            n = affine_descriptors[i].shape[1]
            if n > 1000:  # 防止显存溢出
                return torch.tensor(0., requires_grad=True).to(descriptor_pred), False

            descriptor = descriptor.view(D, -1, 1)
            affine_descriptor = affine_descriptor.view(D, 1, -1)
            ar = torch.arange(n)

            # 随机负样本采样
            neg_index2 = []
            if n == 1:
                neg_index2.append(0)
            else:
                for j in range(n):
                    t = j
                    while t == j:
                        t = random.randint(0, n - 1)
                    neg_index2.append(t)
            neg_index2 = torch.tensor(neg_index2, dtype=torch.long).to(affine_descriptor)

            # 最难负样本采样 (Hard Negative Mining)
            with torch.no_grad():
                dis = torch.norm(descriptor - affine_descriptor, dim=0)
                dis[ar, ar] = dis.max() + 1 # 排除正样本自身
                neg_index1 = dis.argmin(axis=1)

            positive.append(affine_descriptor[:, 0, :].permute(1, 0))
            anchor.append(descriptor[:, :, 0].permute(1, 0))
            negatives_hard.append(affine_descriptor[:, 0, neg_index1.long(), ].permute(1, 0))
            negatives_random.append(affine_descriptor[:, 0, neg_index2.long(), ].permute(1, 0))

        if len(positive) == 0:
            return torch.tensor(0., requires_grad=True).to(descriptor_pred), False

        # 拼接所有样本并进行三元组损失计算
        positive = torch.cat(positive)
        anchor = torch.cat(anchor)
        negatives_hard = torch.cat(negatives_hard)
        negatives_random = torch.cat(negatives_random)

        positive = F.normalize(positive, dim=-1, p=2)
        anchor = F.normalize(anchor, dim=-1, p=2)
        negatives_hard = F.normalize(negatives_hard, dim=-1, p=2)
        negatives_random = F.normalize(negatives_random, dim=-1, p=2)

        # 使用改进的跨模态三元组损失
        loss = triplet_margin_loss_gor(anchor, positive, negatives_hard, negatives_random, margin=0.8)
        
        return loss, True

    def descriptor_loss_warmup(self, label_point_positions, descriptor_pred_fix,
                               descriptor_pred_mov, H_0to1, max_points_per_img=512):
        """
        描述子热身阶段的损失 (改进版: 跨图像负样本采样):
        使用数据集中提供的真值仿射矩阵 H_0to1, 将 CF 上的 vessel_keypoints
        映射到 moving 图像上, 直接构造跨模态正样本对, 再进行三元组损失。
        
        改进: 负样本不仅来自同一张 moving 图像，还包括 batch 内其他图像的点。
        这迫使模型产生全局可区分的描述子，防止描述子坍塌。
        """
        device = descriptor_pred_fix.device
        B, _, H, W = label_point_positions.shape
        D = descriptor_pred_fix.shape[1]

        # 将 H_0to1 转到当前设备, 形状应为 [B, 3, 3]
        if H_0to1 is None:
            # 没有提供真值几何, 返回 0 loss, 但保持梯度图结构
            return torch.tensor(0., requires_grad=True, device=device), False

        if H_0to1.dim() == 2:
            H_0to1 = H_0to1.unsqueeze(0)
        H_0to1 = H_0to1.to(device)

        # ===== 第一阶段: 收集所有样本的描述子 =====
        all_desc_fix = []  # 每个样本的 fix 描述子列表 [N_b, D]
        all_desc_mov = []  # 每个样本的 mov 描述子列表 [N_b, D]
        sample_sizes = []  # 每个样本的点数

        for b in range(B):
            # 取出该样本的关键点掩码 [H, W]
            kp_mask = (label_point_positions[b, 0] > 0.5)
            ys, xs = torch.where(kp_mask)
            if ys.numel() == 0:
                all_desc_fix.append(None)
                all_desc_mov.append(None)
                sample_sizes.append(0)
                continue

            # 随机下采样, 避免显存过大
            if ys.numel() > max_points_per_img:
                idx = torch.randperm(ys.numel(), device=device)[:max_points_per_img]
                ys = ys[idx]
                xs = xs[idx]

            # 构造齐次坐标并应用 H_0to1, 得到 moving 图像上的对应点
            ones = torch.ones_like(xs, dtype=torch.float32, device=device)
            pts_fix = torch.stack([xs.float(), ys.float(), ones], dim=0)  # [3, N]
            H_mat = H_0to1[b]  # [3, 3] 真值仿射矩阵
            pts_mov = H_mat @ pts_fix  # [3, N]

            # 归一化 (一般仿射, pts_mov[2]≈1)
            xs_mov = pts_mov[0] / (pts_mov[2] + 1e-6)
            ys_mov = pts_mov[1] / (pts_mov[2] + 1e-6)

            # 只保留在图像内部的点
            valid = (xs_mov >= 0) & (xs_mov <= (W - 1)) & (ys_mov >= 0) & (ys_mov <= (H - 1))
            if valid.sum() == 0:
                all_desc_fix.append(None)
                all_desc_mov.append(None)
                sample_sizes.append(0)
                continue

            xs_fix_valid = xs[valid]
            ys_fix_valid = ys[valid]
            xs_mov_valid = xs_mov[valid]
            ys_mov_valid = ys_mov[valid]

            # 组装成 [N, 2] 的 (x, y) 坐标
            kps_fix = torch.stack([xs_fix_valid, ys_fix_valid], dim=1)  # [N, 2]
            kps_mov = torch.stack([xs_mov_valid, ys_mov_valid], dim=1)  # [N, 2]

            # 基于坐标从 descriptor feature map 中采样描述子
            desc_fix = sample_keypoint_desc(kps_fix[None], descriptor_pred_fix[b:b+1], s=self.scale)[0]   # [C, N]
            desc_mov = sample_keypoint_desc(kps_mov[None], descriptor_pred_mov[b:b+1], s=self.scale)[0]   # [C, N]

            n = desc_fix.shape[1]
            if n == 0 or n > 1000:
                all_desc_fix.append(None)
                all_desc_mov.append(None)
                sample_sizes.append(0)
                continue

            all_desc_fix.append(desc_fix.permute(1, 0))  # [N, D]
            all_desc_mov.append(desc_mov.permute(1, 0))  # [N, D]
            sample_sizes.append(n)

        # ===== 第二阶段: 构建跨图像负样本池 =====
        # 将所有有效的 moving 描述子拼接成一个大池子
        valid_mov_list = [d for d in all_desc_mov if d is not None]
        if len(valid_mov_list) == 0:
            return torch.tensor(0., requires_grad=True, device=device), False
        
        all_mov_pool = torch.cat(valid_mov_list, dim=0)  # [Total_N, D]
        total_pool_size = all_mov_pool.shape[0]

        # ===== 第三阶段: 为每个 anchor 采样负样本 =====
        anchor_list = []
        positive_list = []
        negatives_hard_list = []
        negatives_cross_list = []  # 跨图像负样本

        # 计算每个样本在 pool 中的起始索引
        pool_offsets = [0]
        for size in sample_sizes:
            pool_offsets.append(pool_offsets[-1] + size)

        for b in range(B):
            if all_desc_fix[b] is None:
                continue
            
            desc_fix = all_desc_fix[b]  # [N, D]
            desc_mov = all_desc_mov[b]  # [N, D]
            n = desc_fix.shape[0]
            
            # 当前样本在 pool 中的范围
            start_idx = pool_offsets[b]
            end_idx = pool_offsets[b + 1]
            
            # ----- 同图像内硬负样本 (保留原有逻辑) -----
            with torch.no_grad():
                dis = torch.cdist(desc_fix, desc_mov, p=2)  # [N, N]
                ar = torch.arange(n, device=device)
                dis[ar, ar] = dis.max() + 1  # 排除正样本本身
                neg_index_hard = dis.argmin(dim=1)
            
            # ----- 跨图像负样本 (从整个 pool 中采样，排除当前样本) -----
            # 为每个 anchor 随机采样一个来自其他图像的负样本
            neg_index_cross = []
            for j in range(n):
                # 从 pool 中排除当前样本的范围
                if total_pool_size == n:
                    # 只有一个样本，退回到同图像采样
                    neg_idx = random.randint(0, n - 1)
                    if neg_idx == j:
                        neg_idx = (neg_idx + 1) % n
                    neg_index_cross.append(neg_idx + start_idx)
                else:
                    # 从其他样本中采样
                    while True:
                        idx = random.randint(0, total_pool_size - 1)
                        if idx < start_idx or idx >= end_idx:
                            break
                    neg_index_cross.append(idx)
            neg_index_cross = torch.tensor(neg_index_cross, dtype=torch.long, device=device)

            anchor_list.append(desc_fix)
            positive_list.append(desc_mov)
            negatives_hard_list.append(desc_mov[neg_index_hard])
            negatives_cross_list.append(all_mov_pool[neg_index_cross])

        if len(anchor_list) == 0:
            return torch.tensor(0., requires_grad=True, device=device), False

        anchor = torch.cat(anchor_list, dim=0)
        positive = torch.cat(positive_list, dim=0)
        negatives_hard = torch.cat(negatives_hard_list, dim=0)
        negatives_cross = torch.cat(negatives_cross_list, dim=0)

        # 归一化后使用三元组损失
        anchor = F.normalize(anchor, dim=-1, p=2)
        positive = F.normalize(positive, dim=-1, p=2)
        negatives_hard = F.normalize(negatives_hard, dim=-1, p=2)
        negatives_cross = F.normalize(negatives_cross, dim=-1, p=2)

        # 使用跨图像负样本替换原来的随机负样本，增强全局判别力
        loss = triplet_margin_loss_gor(anchor, positive, negatives_hard, negatives_cross, margin=0.8)
        return loss, True
    
    def forward(self, fix_img, mov_img, label_point_positions=None, value_map=None, learn_index=None,
                phase=3, vessel_mask=None, H_0to1=None):
        """
        主前向传播逻辑 - 支持四阶段训练策略
        :param phase: 训练阶段 (1: Warmup, 2: Geo-Consistency, 3: PKE Joint Training)
        :param fix_img: 固定图像 (CF)
        :param mov_img: 运动图像 (FA/OCT)
        :param label_point_positions: GT 关键点标签 (Phase 1/2 使用)
        :param H_0to1: 真值单应性矩阵 (Phase 1/2 使用)
        """
        
        # 1. 提取固定图像（CF 模态）与运动图像的特征
        detector_pred_fix, descriptor_pred_fix = self.network(fix_img)
        detector_pred_mov, descriptor_pred_mov = self.network(mov_img)
        
        enhanced_label_pts = None
        enhanced_label = None
        loss_detector_num = 0
        loss_descriptor_num = 0

        # 推断模式
        if label_point_positions is None:
             return detector_pred_fix, descriptor_pred_fix

        # 训练模式准备
        B, _, H, W = detector_pred_fix.shape
        loss_detector = torch.tensor(0., requires_grad=True).to(fix_img)
        loss_descriptor = torch.tensor(0., requires_grad=True).to(fix_img)
        number_pts = 0

        # =========================================================
        # Phase 1 & 2: 热身期 & 几何一致性预热 (PKE OFF)
        # =========================================================
        if phase in [1, 2]:
            loss_descriptor_num = B
            
            # --- A. 描述子损失 (全程) ---
            # 使用真值 H_0to1 构造几何关系，无需额外 Image Augmentation
            # 注意：Phase 1/2 我们只信任 GT H 带来的对应关系
            loss_descriptor, _ = self.descriptor_loss_warmup(
                label_point_positions, descriptor_pred_fix, descriptor_pred_mov, H_0to1
            )
            
            # --- B. Fix 检测器热身 (Phase 1 & 2) ---
            # 强监督：利用 GT 分叉点生成的高斯热力图训练 Fix Detector
            # label_point_positions 是 0/1 mask，需要平滑为 soft label
            gt_heatmap = F.conv2d(label_point_positions, self.kernel, 
                                  stride=1, padding=(self.kernel.shape[-1] - 1) // 2)
            gt_heatmap[gt_heatmap > 1] = 1.0
            
            # loss_det_fix = Dice(pred, gt)
            loss_det_warm = self.dice(detector_pred_fix, gt_heatmap)
            loss_detector = loss_detector + loss_det_warm
            
            # --- C. Moving 检测器几何对齐 (Phase 2 Only) ---
            # 利用 H_0to1 将 Moving 检测图 warp 回 Fix 空间，要求其与 Fix 检测图一致
            if phase == 2:
                if H_0to1 is not None:
                     # 构造 grid 用于 warp (moving -> fix, 即 H^-1)
                    if H_0to1.dim() == 2: H_0to1 = H_0to1.unsqueeze(0)
                    
                    # 求逆矩阵: H_fix->mov = H_0to1 => H_mov->fix = inv(H_0to1)
                    try:
                        H_inv = torch.inverse(H_0to1)
                    except:
                        H_inv = torch.eye(3, device=fix_img.device).repeat(B, 1, 1) # Fallback

                    # 构造采样网格
                    grid_list = []
                    ys, xs = torch.meshgrid(torch.arange(H, device=fix_img.device), torch.arange(W, device=fix_img.device), indexing='ij')
                    # 归一化坐标系构造 grid 比较繁琐，这里使用 affine_grid 的简化版逻辑
                    # 直接构造像素坐标 -> 变换 -> 归一化
                    ones = torch.ones_like(xs, dtype=torch.float32)
                    pts = torch.stack([xs.float(), ys.float(), ones], dim=-1).view(-1, 3) # (N, 3)
                    
                    for b in range(B):
                        # H_inv * pts_fix = pts_mov (找到 fix 像素对应的 moving 坐标)
                        h_mat = H_0to1[b] # 这里要用 H_0to1 (fix -> moving) 还是 inverse?
                        # F.grid_sample(input, grid) 中 grid 对于输出点 (x,y)，采样 input 在 (u,v) 的值
                        # 我们想要输出 warped_det_mov (在 fix 坐标系)，所以对于 fix 上的点 (x,y)，
                        # 我们需要知道它在 moving 图上的位置 (u,v) = H_0to1 * (x,y)
                        
                        # 所以这里确实是用 H_0to1 直接算映射坐标
                        pts_mov = pts @ h_mat.t() 
                        u = pts_mov[:, 0] / (pts_mov[:, 2] + 1e-6)
                        v = pts_mov[:, 1] / (pts_mov[:, 2] + 1e-6)
                        
                        u = 2.0 * u / (W - 1) - 1.0
                        v = 2.0 * v / (H - 1) - 1.0
                        
                        grid_list.append(torch.stack([u, v], dim=-1).view(H, W, 2))
                        
                    grid = torch.stack(grid_list, dim=0) # [B, H, W, 2]
                    
                    # 采样：warp moving detection to fix coordinate
                    det_mov_warped = F.grid_sample(detector_pred_mov, grid, align_corners=True)
                    
                    # 几何一致性 Loss: Dice(det_fix, det_mov_warped)
                    # 系数设为 0.5 (可调)
                    loss_geo_warm = self.dice(detector_pred_fix, det_mov_warped)
                    loss_detector = loss_detector + 0.5 * loss_geo_warm

            loss_detector_num = B # 或者是别的 counting 逻辑，暂用 B
            
            return loss_detector + loss_descriptor, number_pts, loss_detector.detach().sum(), \
                   loss_descriptor.detach().sum(), None, None, detector_pred_fix, loss_detector_num, loss_descriptor_num

        # =========================================================
        # Phase 3: PKE 联合训练 (Joint Training)
        # =========================================================
        elif phase == 3:
            loss_detector_num = len(learn_index[0])
            loss_descriptor_num = fix_img.shape[0]

            # 准备 Grid Inverse (用于 PKE 内部的 mapping_points)
            # 这里的 grid_inverse 应该是: 给定 moving 上的点，找 fix 上的对应点
            # 即 H_inv = H_0to1^-1
            # 原代码逻辑：grid_inverse 用于将 fix 坐标映射到 moving 坐标（这就叫 inverse? 只有看完 context 才知道）
            # Check pke_module.mapping_points: grid_inverse, points (on fix) -> output (on moving)
            # 所以 pke_module 里需要的 grid 是 "Fix -> Moving" 的映射
            
            # 复用 Phase 1 里的 grid 构建逻辑 (H_0to1: Fix -> Moving)
            if H_0to1 is not None:
                if H_0to1.dim() == 2: H_0to1 = H_0to1.unsqueeze(0)
                grid_list = []
                ys, xs = torch.meshgrid(torch.arange(H, device=fix_img.device), torch.arange(W, device=fix_img.device), indexing='ij')
                pts = torch.stack([xs.float(), ys.float(), torch.ones_like(xs, dtype=torch.float32)], dim=-1).view(-1, 3)
                
                for b in range(B):
                    # Fix -> Moving 映射
                    pts_mov = pts @ H_0to1[b].t()
                    u = pts_mov[:, 0] / (pts_mov[:, 2] + 1e-6)
                    v = pts_mov[:, 1] / (pts_mov[:, 2] + 1e-6)
                    u = 2.0 * u / (W - 1) - 1.0
                    v = 2.0 * v / (H - 1) - 1.0
                    grid_list.append(torch.stack([u, v], dim=-1).view(H, W, 2))
                grid_inverse = torch.stack(grid_list, dim=0)
            else:
                 # Fallback (should not happen in this dataset)
                 grid_inverse = torch.zeros(B, H, W, 2).to(fix_img.device)

            # PKE 核心过程
            value_map_update = None
            if len(learn_index[0]) != 0:
                loss_detector, number_pts, value_map_update, enhanced_label_pts, enhanced_label = \
                    pke_learn(detector_pred_fix[learn_index], descriptor_pred_fix[learn_index],
                              grid_inverse[learn_index], detector_pred_mov[learn_index],
                              descriptor_pred_mov[learn_index], self.kernel, self.dice,
                              label_point_positions[learn_index], value_map[learn_index],
                              self.config, PKE_learn=True, vessel_mask=vessel_mask[learn_index] if vessel_mask is not None else None)
            
            # 更新 Value Map
            if value_map is not None and value_map_update is not None:
                value_map[learn_index] = value_map_update

            # 联合期描述子 Loss (自监督 + GT辅助)
            # 为了稳健，我们可以继续保留 descriptor_loss_warmup 的约束 (GT H)，
            # 也可以切回完全自监督。Plan v5 建议是 "Des Self"，但其实如果有 GT H，用 GT H 采样的 triplet 永远是最准的。
            # 这里我们还是用 descriptor_loss_warmup (底层逻辑一样，都是 Triplet)，因为它利用了真值 H。
            # 只要 grid 准，triplet 就准。
            loss_descriptor, _ = self.descriptor_loss_warmup(
                label_point_positions, descriptor_pred_fix, descriptor_pred_mov, H_0to1
            )
            
            return loss_detector + loss_descriptor, number_pts, loss_detector.detach().sum(), \
                   loss_descriptor.detach().sum(), enhanced_label_pts, \
                   enhanced_label, detector_pred_fix, loss_detector_num, loss_descriptor_num

        return torch.tensor(0.), 0, 0, 0, None, None, None, 0, 0
