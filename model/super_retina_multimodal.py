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

    def forward(self, fix_img, mov_img, label_point_positions=None, value_map=None, learn_index=None, descriptor_only=False, vessel_mask=None):
        """
        主前向传播逻辑
        :param fix_img: 固定图像 (CF)
        :param mov_img: 运动图像 (FA/OCT)，训练时与 fix_img 对齐
        :param label_point_positions: 初始种子点标签
        :param descriptor_only: 若为True，则仅训练描述子（跳过检测器损失和PKE）
        :param vessel_mask: 固定图像的血管分割图 [B, 1, H, W]，用于PKE候选点过滤
        """
        
        # 1. 提取固定图像（CF 模态）的特征
        detector_pred_fix, descriptor_pred_fix = self.network(fix_img)
        
        enhanced_label_pts = None
        enhanced_label = None

        if label_point_positions is not None: # 训练模式
            if self.PKE_learn and not descriptor_only:
                loss_detector_num = len(learn_index[0])
                loss_descriptor_num = fix_img.shape[0]
            else:
                loss_detector_num = len(learn_index[0])
                loss_descriptor_num = loss_detector_num

            number_pts = 0 
            value_map_update = None
            loss_detector = torch.tensor(0., requires_grad=True).to(fix_img)
            loss_descriptor = torch.tensor(0., requires_grad=True).to(fix_img)

            # 2. 对运动图像（FA/OCT 模态）进行几何扰动 (Data Augmentation)
            # 模拟真实配准中的大尺度旋转和平移
            with torch.no_grad():
                affine_mov, grid, grid_inverse = affine_images(mov_img, used_for='detector')
                
                # 有效区域掩码：排除变换产生的黑色边缘
                valid_mask = (affine_mov > 0.05).float()
                # 腐蚀操作，确保关键点不在边缘采样
                valid_mask = -F.max_pool2d(-valid_mask, kernel_size=5, stride=1, padding=2)
            
            # 提取扰动后运动图像的特征
            detector_pred_mov_aug, descriptor_pred_mov_aug = self.network(affine_mov)
            
            # 屏蔽边缘区域的检测响应
            detector_pred_mov_aug = detector_pred_mov_aug * valid_mask

            # 3. 跨模态渐进式关键点扩充 (PKE)
            # 核心思想：通过几何校验和内容校验，自动发现两模态间可靠的匹配点作为新增标签
            loss_cal = self.dice
            if len(learn_index[0]) != 0 and not descriptor_only:
                # pke_learn 内部实现了几何一致性损失 (l_geo) 和 动态标签演化 (l_clf)
                loss_detector, number_pts, value_map_update, enhanced_label_pts, enhanced_label = \
                    pke_learn(detector_pred_fix[learn_index], descriptor_pred_fix[learn_index],
                              grid_inverse[learn_index], detector_pred_mov_aug[learn_index],
                              descriptor_pred_mov_aug[learn_index], self.kernel, loss_cal,
                              label_point_positions[learn_index], value_map[learn_index],
                              self.config, self.PKE_learn, vessel_mask=vessel_mask[learn_index] if vessel_mask is not None else None)

            # 辅助可视化
            if enhanced_label_pts is not None:
                enhanced_label_pts_tmp = label_point_positions.clone()
                enhanced_label_pts_tmp[learn_index] = enhanced_label_pts
                enhanced_label_pts = enhanced_label_pts_tmp
            if enhanced_label is not None:
                enhanced_label_tmp = label_point_positions.clone()
                enhanced_label_tmp[learn_index] = enhanced_label
                enhanced_label = enhanced_label_tmp

            detector_pred_copy = detector_pred_fix.clone().detach()

            # 4. 跨模态描述子损失 (l_des)
            # 再次生成轻微扰动的运动图像，用于描述子学习（通常比检测器训练的扰动更小，以学习精细特征）
            affine_mov_desc, grid_desc, grid_inverse_desc = affine_images(mov_img, used_for='descriptor')
            _, descriptor_pred_mov_aug_desc = self.network(affine_mov_desc)
            
            loss_descriptor, descriptor_train_flag = self.descriptor_loss(
                detector_pred_copy, label_point_positions,
                descriptor_pred_fix,
                descriptor_pred_mov_aug_desc,
                grid_inverse_desc,
                affine_detector_pred=None 
            )

            if self.PKE_learn and len(learn_index[0]) != 0:
                value_map[learn_index] = value_map_update
            
            # 综合损失优化
            loss = loss_detector + loss_descriptor

            return loss, number_pts, loss_detector.cpu().data.sum(), \
                   loss_descriptor.cpu().data.sum(), enhanced_label_pts, \
                   enhanced_label, detector_pred_fix, loss_detector_num, loss_descriptor_num

        # 推断阶段：仅输出固定图像的检测图和描述子
        return detector_pred_fix, descriptor_pred_fix
