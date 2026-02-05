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


# ============================================================================
# InfoNCE Loss (对比学习损失)
# 相比 Triplet Loss 使用 batch 内所有其他点作为负样本, 提供更强的对比信号
# 可有效防止描述子坍塌 (Descriptor Collapse)
# ============================================================================

def info_nce_loss(anchor, positive, all_negatives, temperature=0.07):
    """
    InfoNCE Loss (Noise Contrastive Estimation)
    
    Args:
        anchor: [N, D] 锚点描述子 (来自 fix 图像)
        positive: [N, D] 正样本描述子 (来自 moving 图像，与 anchor 一一对应)
        all_negatives: [M, D] 负样本池 (batch 内所有 moving 描述子)
        temperature: 温度参数, 控制分布的锐利程度, 越小越尖锐
    
    Returns:
        loss: InfoNCE 损失值
    """
    N = anchor.shape[0]
    
    # 计算 anchor 与 positive 的相似度 (正样本)
    # [N, D] x [N, D] -> [N] (对应位置点积)
    pos_sim = (anchor * positive).sum(dim=1) / temperature  # [N]
    
    # 计算 anchor 与所有负样本的相似度
    # [N, D] x [M, D].T -> [N, M]
    neg_sim = torch.mm(anchor, all_negatives.T) / temperature  # [N, M]
    
    # 将正样本相似度插入到第一个位置，构成 [N, M+1] 的 logits
    # 这样 target 就是全 0 向量 (正样本在位置 0)
    logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)  # [N, M+1]
    
    # 目标是让第 0 个位置 (正样本) 的概率最大
    targets = torch.zeros(N, dtype=torch.long, device=anchor.device)
    
    # 使用交叉熵损失
    loss = F.cross_entropy(logits, targets)
    
    return loss




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


class SuperRetinaEncoder(nn.Module):
    """
    Independent Encoder Module for SuperRetina
    """
    def __init__(self, c1=64, c2=64, c3=128, c4=128):
        super().__init__()
        self.relu = torch.nn.ReLU(inplace=True)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv1a = torch.nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
        self.conv1b = torch.nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
        self.conv2a = torch.nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.conv2b = torch.nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
        self.conv3a = torch.nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.conv3b = torch.nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
        self.conv4a = torch.nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        self.conv4b = torch.nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
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
        return x, conv1, conv2, conv3

class SuperRetinaMultimodal(nn.Module):
    """
    SuperRetina 多模态配准网络
    v8 版本：CF-Anchor 双路多模态对齐架构 (无 PKE)
    """
    def __init__(self, config=None, device='cpu', n_class=1):
        super().__init__()

        self.PKE_learn = False # v8 取消 PKE
        self.relu = torch.nn.ReLU(inplace=True)
        c1, c2, c3, c4, c5, d1, d2 = 64, 64, 128, 128, 256, 256, 256
        
        # 配准策略配置
        self.shared_encoder = True
        if config is not None:
             self.shared_encoder = config.get('shared_encoder', True)

        # --- 编码器 (Encoder) ---
        if self.shared_encoder:
            # 共享权重 (Siamese)
            self.encoder = SuperRetinaEncoder(c1, c2, c3, c4)
        else:
            # 双路编码器 (Pseudo-Siamese)
            # encoder_fix: 处理固定图像 (CF, Dark Vessels)
            # encoder_mov: 处理运动图像 (FA/OCTA, Bright Vessels)
            self.encoder_fix = SuperRetinaEncoder(c1, c2, c3, c4)
            self.encoder_mov = SuperRetinaEncoder(c1, c2, c3, c4)

        # --- 描述子头部 (Descriptor Head) --- (共享)
        self.convDa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convDb = torch.nn.Conv2d(c5, d1, kernel_size=4, stride=2, padding=0)
        self.convDc = torch.nn.Conv2d(d1, d2, kernel_size=1, stride=1, padding=0)
        self.trans_conv = nn.ConvTranspose2d(d1, d2, 2, stride=2)

        # --- 检测器头部 (Detector Head) --- (共享)
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
            kernel_size = config.get('gaussian_kernel_size', 21)
            sigma = config.get('gaussian_sigma', 3)
            self.kernel = get_gaussian_kernel(kernlen=kernel_size, nsig=sigma).to(device)

        self.to(device)

    def network(self, x, mode='fix'):
        """
        前向传播基础网络：提取检测图 P 和描述子张量 D
        :param x: 输入图像
        :param mode: 'fix' or 'mov' (仅在 shared_encoder=False 时生效)
        """
        # 编码阶段
        if self.shared_encoder:
            x, conv1, conv2, conv3 = self.encoder(x)
        else:
            if mode == 'fix':
                x, conv1, conv2, conv3 = self.encoder_fix(x)
            else:
                x, conv1, conv2, conv3 = self.encoder_mov(x)

        # --- 描述子分支 ---
        cDa = self.relu(self.convDa(x))
        cDb = self.relu(self.convDb(cDa))
        desc = self.convDc(cDb)
        dn = torch.norm(desc, p=2, dim=1)
        desc = desc.div(torch.unsqueeze(dn, 1))
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
        semi = torch.sigmoid(semi)

        return semi, desc

    def descriptor_loss(self, detector_pred, label_point_positions, descriptor_pred_map,
                        affine_descriptor_pred, grid_inverse, affine_detector_pred=None):
        """
        计算描述子损失 (Triplet Loss)
        强制相同解剖点在不同模态下的特征描述子尽可能接近
        """
        # 核心修复：克隆 detector_pred，防止对原始梯度张量的原地修改导致 Autograd 报错
        detector_pred_samp = detector_pred.clone()

        if not self.PKE_learn:
            detector_pred_samp[:] = 0  # 冷启动阶段仅使用初始种子点
        # 将初始标签点权权重大幅提升，确保它们被选中进行匹配
        detector_pred_samp[label_point_positions == 1] = 10
        
        # 采样关键点及其对应的描述子
        descriptors, affine_descriptors, keypoints = \
            sample_descriptors(detector_pred_samp, descriptor_pred_map, affine_descriptor_pred, grid_inverse,
                               nms_size=self.nms_size, nms_thresh=self.nms_thresh, scale=self.scale,
                               affine_detector_pred=affine_detector_pred)

        positive = []
        negatives_hard = []
        negatives_random = []
        anchor = []
        D = descriptor_pred_map.shape[1]
        
        for i in range(len(affine_descriptors)):
            if affine_descriptors[i].shape[1] == 0:
                continue
            descriptor = descriptors[i]
            affine_descriptor = affine_descriptors[i]

            n = affine_descriptors[i].shape[1]
            if n > 1000:  # 防止显存溢出
                return torch.tensor(0., requires_grad=True).to(detector_pred_samp), False

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
            return torch.tensor(0., requires_grad=True).to(detector_pred_samp), False

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

    def dense_alignment_loss(self, desc_fix, desc_mov, vessel_mask, num_samples=1024, temperature=0.07):
        """
        v6.2 Phase 0: 密集对齐损失 (Efficient Dense InfoNCE)
        目标: 在解剖结构完全一致的情况下，强制 Moving 分支模仿 Fix 分支的特征分布。
        策略: 分层采样 (50% 血管, 50% 随机) + 样本内对比。
        """
        device = desc_fix.device
        B, C, H_feat, W_feat = desc_fix.shape
        
        # 处理无掩码情况：如果 vessel_mask 为 None，则默认全图为 1
        if vessel_mask is not None:
            # 将 Mask 下采样到特征图尺寸 (通常是 1/8)
            mask_feat = F.interpolate(vessel_mask, size=(H_feat, W_feat), mode='nearest')
        else:
            mask_feat = torch.ones((B, 1, H_feat, W_feat), device=desc_fix.device)
            
        total_loss = 0.0
        for b in range(B):
            # 1. 提取当前样本的描述子图并展平
            feat_fix = desc_fix[b].view(C, -1).permute(1, 0) # [N_total, C]
            feat_mov = desc_mov[b].view(C, -1).permute(1, 0) # [N_total, C]
            m = mask_feat[b, 0].view(-1) # [N_total]
            
            # 2. 分层采样正样本索引 (Anchors)
            vessel_indices = torch.where(m > 0.5)[0]
            all_indices = torch.arange(H_feat * W_feat, device=device)
            
            # --- 50% 血管采样 ---
            if vessel_indices.numel() > (num_samples // 2):
                idx_v = vessel_indices[torch.randperm(vessel_indices.numel())[:num_samples // 2]]
            else:
                idx_v = vessel_indices # 血管点全取
            
            # --- 50% 随机全图采样 ---
            idx_r = all_indices[torch.randperm(all_indices.numel())[:num_samples - idx_v.numel()]]
            
            # 组合采样点索引
            idx = torch.cat([idx_v, idx_r])
            
            # 3. 构造正负样本对
            # 让双路同时训练，共同构建高区分度的跨模态特征空间
            anchor = F.normalize(feat_fix[idx], dim=-1)   # [N, C] (CF 特征)
            positive = F.normalize(feat_mov[idx], dim=-1) # [N, C] (FA/OCTA 特征)
            
            # 负样本池: 使用当前图内所有的 Moving 特征 (或者更高效地只用当前图的特征)
            # 为了极端效率，我们只用当前图的特征作为负样本库
            pool_mov = F.normalize(feat_mov, dim=-1) # [N_total, C]
            
            # 4. 计算 InfoNCE (密集矩阵运算)
            # 计算 Similarity: [N, C] @ [C, N_total] -> [N, N_total]
            logits = torch.mm(anchor, pool_mov.t()) / temperature 
            
            # 对应的正样本索引即为 idx 在 pool 中的位置
            # 由于 pool_mov 是按顺序展平的，所以 idx 就是对应的位置
            targets = idx 
            
            loss = F.cross_entropy(logits, targets)
            total_loss += loss
            
        return total_loss / B
    
    def construct_grid_from_H(self, H_mat, H, W):
        """
        根据单应矩阵/仿射矩阵构建 grid_sample 所需的 grid
        H_mat: [B, 3, 3]  (Fix -> Mov, x' = Hx)
        Return: [B, H, W, 2] in [-1, 1] range
        """
        B = H_mat.shape[0]
        device = H_mat.device
        
        # 生成归一化坐标网格
        ys, xs = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
        # [H, W, 3] Homogeneous coords
        pts = torch.stack([xs.float(), ys.float(), torch.ones_like(xs, dtype=torch.float32)], dim=-1).view(-1, 3)
        
        grid_list = []
        for b in range(B):
            # Fix -> Mov 变换: pts_mov = H @ pts_fix^T
            # [3, 3] @ [3, N] -> [3, N] -> [N, 3]
            pts_mov = (H_mat[b] @ pts.t()).t()
            
            # 归一化到 [-1, 1]
            # u = 2 * x / (W-1) - 1
            # v = 2 * y / (H-1) - 1
            # 注意处理透视除法 (虽然仿射通常是 1)
            z = pts_mov[:, 2:3] + 1e-6
            u = (2.0 * (pts_mov[:, 0:1] / z) / (W - 1)) - 1.0
            v = (2.0 * (pts_mov[:, 1:2] / z) / (H - 1)) - 1.0
            
            grid_list.append(torch.cat([u, v], dim=-1).view(H, W, 2))
            
        return torch.stack(grid_list, dim=0)

    def forward_v8(self, fix_img, mov_img, H_0to1, vessel_mask_fix, lambda_desc=1.0, lambda_det=0.5):
        """
        SuperRetina v8 前向传播：CF 锚点密集对齐 + 检测器蒸馏 (无 PKE)
        
        参数:
            fix_img: [B, 1, H, W] CF 图像 (作为锚点分支)
            mov_img: [B, 1, H, W] FA/OCT 图像 (做为运动分支/训练分支)
            H_0to1: [B, 3, 3] 真值变换矩阵 (Fix -> Mov)
            vessel_mask_fix: [B, 1, H, W] CF 上的血管掩膜 (用于限制对齐区域)
            lambda_desc: 描述子损失权重
            lambda_det: 检测器蒸馏损失权重
        """
        # 1. 基础网络前向传播
        # CF 图像始终走 encoder_fix (冻结状态)，FA/OCT 图像走 encoder_mov (训练状态)
        det_fix, desc_fix = self.network(fix_img, mode='fix')
        det_mov, desc_mov = self.network(mov_img, mode='mov')
        
        B, C, H_feat, W_feat = desc_fix.shape
        _, _, H_img, W_img = fix_img.shape
        
        # 2. 构建空间变换网格 (Fix -> Mov)
        # 用于将 Moving 端的特征图和检测图采样回 CF 坐标系，实现像素级的空间对齐
        grid_feat = self.construct_grid_from_H(H_0to1, H_feat, W_feat) # 特征图尺度
        grid_img = self.construct_grid_from_H(H_0to1, H_img, W_img)    # 原始图像尺度
        
        # 3. 描述子稠密对齐损失 (L_desc)
        # 将 Moving 特征场 warp 回 CF 坐标系
        desc_mov_warped = F.grid_sample(desc_mov, grid_feat, mode='bilinear', align_corners=True)
        
        # 血管区域掩膜处理：将血管掩膜下采样到特征图尺度
        if vessel_mask_fix is not None:
            mask_feat = F.interpolate(vessel_mask_fix, size=(H_feat, W_feat), mode='nearest')
            # 阈值化，确定有效的血管区域
            mask_valid = (mask_feat > 0.5).float()
        else:
            # 如果没有血管掩膜，则在全图进行对齐
            mask_valid = torch.ones_like(desc_fix[:, 0:1, :, :])
            
        # 计算特征间的 MSE 损失 (L2 距离)
        # 目标：让 encoder_mov 在血管区域学到与 encoder_fix 相同的结构特征分布
        diff = desc_fix - desc_mov_warped
        loss_desc_map = (diff ** 2).sum(dim=1, keepdim=True) # [B, 1, H_feat, W_feat]
        
        # 加权平均：只在有效血管区域计算损失
        loss_desc = (loss_desc_map * mask_valid).sum() / (mask_valid.sum() + 1e-6)
        
        # 4. 检测器蒸馏损失 (L_det)
        # 将 Moving 端预测的检测图 warp 回 CF 坐标系
        det_mov_warped = F.grid_sample(det_mov, grid_img, mode='bilinear', align_corners=True)
        
        # 蒸馏目标：Moving 端在 warp 后应与 CF 端产生的检测图相似
        # det_fix 视为软标签 (detach 掉以防止其参数被更新)，使用 Dice Loss 进行约束
        loss_det = self.dice(det_mov_warped, det_fix.detach())
        
        # 5. 总损失计算
        loss_total = lambda_desc * loss_desc + lambda_det * loss_det
        
        return loss_total, loss_desc, loss_det, det_fix, det_mov_warped

