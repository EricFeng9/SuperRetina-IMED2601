import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F

from common.common_util import sample_keypoint_desc, nms
from model.record_module import update_value_map


def mapping_points(grid, points, h, w):
    """ 
    利用逆变换网格将固定图像上的坐标映射到变换后的运动图像坐标系中
    :param grid: 逆变换网格 (grid_inverse)，从运动图像映射回固定图像
    :param points: 固定图像上的关键点坐标
    :return: 经过筛选的原始点集和对应的变换后点集
    """
    # 获取网格中对应位置的坐标值
    grid_points = [(grid[s, k[:, 1].long(), k[:, 0].long()]) for s, k in
                   enumerate(points)]
    filter_points = []
    affine_points = []
    for s, k in enumerate(grid_points):  # 过滤超出图像范围的点
        idx = (k[:, 0] < 1) & (k[:, 0] > -1) & (k[:, 1] < 1) & (
                k[:, 1] > -1)
        gp = grid_points[s][idx]
        # 将 [-1, 1] 的归一化坐标转换为像素坐标
        gp[:, 0] = (gp[:, 0] + 1) / 2 * (w - 1)
        gp[:, 1] = (gp[:, 1] + 1) / 2 * (h - 1)
        affine_points.append(gp)
        filter_points.append(points[s][idx])

    return filter_points, affine_points


def content_filter(descriptor_pred, affine_descriptor_pred, geo_points,
                   affine_geo_points, content_thresh=0.7, scale=8):
    """
    内容校验 (Content Verification / Double Matching)
    利用描述子间的 Lowe's Ratio Test 筛选跨模态一致的特征点
    """
    # 提取固定图像和运动图像在对应位置的描述子
    descriptors = [sample_keypoint_desc(k[None], d[None], scale)[0].permute(1, 0)
                   for k, d in zip(geo_points, descriptor_pred)]
    aff_descriptors = [sample_keypoint_desc(k[None], d[None], scale)[0].permute(1, 0)
                       for k, d in zip(affine_geo_points, affine_descriptor_pred)]
    
    content_points = []
    affine_content_points = []
    # 计算描述子间的欧氏距离
    dist = [torch.norm(descriptors[d][:, None] - aff_descriptors[d], dim=2, p=2)
            for d in range(len(descriptors))]
    
    for i in range(len(dist)):
        D = dist[i]
        if len(D) <= 1:
            content_points.append([])
            affine_content_points.append([])
            continue
        
        # 获取最近和次近的匹配距离
        val, ind = torch.topk(D, 2, dim=1, largest=False)

        arange = torch.arange(len(D))
        # 准则1：空间对应关系（最近匹配点应当是几何映射后的点）
        c1 = ind[:, 0] == arange.to(ind.device)
        # 准则2：Lowe's Ratio Test (最近距离 / 次近距离 < 阈值)
        c2 = val[:, 0] < val[:, 1] * content_thresh

        check = c2 * c1
        content_points.append(geo_points[i][check])
        affine_content_points.append(affine_geo_points[i][check])
        
    return content_points, affine_content_points


def geometric_filter(affine_detector_pred, points, affine_points, max_num=1024, geometric_thresh=0.5):
    """
    几何校验 (Geometric Verification)
    通过检查运动图像检测器在映射位置的响应值，判断特征点的重现性
    """
    geo_points = []
    affine_geo_points = []
    for s, k in enumerate(affine_points):
        # 采样运动图像检测图在对应坐标处的概率值
        sample_aff_values = affine_detector_pred[s, 0, k[:, 1].long(), k[:, 0].long()]
        # 仅保留响应值大于阈值的点 (即 P > 0.5 且 P' > 0.5)
        check = sample_aff_values.squeeze() >= geometric_thresh
        geo_points.append(points[s][check][:max_num])
        affine_geo_points.append(k[check][:max_num])

    return geo_points, affine_geo_points


def pke_learn(detector_pred, descriptor_pred, grid_inverse, affine_detector_pred,
              affine_descriptor_pred, kernel, loss_cal, label_point_positions,
              value_map, config, PKE_learn=True, vessel_mask=None):
    """
    渐进式关键点扩充 (Progressive Keypoint Expansion, PKE) 核心逻辑
    :param vessel_mask: [B, 1, H, W] 血管分割图，用于过滤候选点
    """
    # 初始种子点标签（由 PBO 提取的血管分叉点等）
    initial_label = F.conv2d(label_point_positions, kernel,
                             stride=1, padding=(kernel.shape[-1] - 1) // 2)
    initial_label[initial_label > 1] = 1

    if not PKE_learn:
        # 冷启动阶段：仅计算针对初始标签的检测损失
        return loss_cal(detector_pred, initial_label.to(detector_pred)), 0, None, None, initial_label

    nms_size = config['nms_size']
    nms_thresh = config['nms_thresh']
    scale = 8

    enhanced_label = None
    geometric_thresh = config['geometric_thresh']
    content_thresh = config['content_thresh']
    
    with torch.no_grad():
        h, w = detector_pred.shape[2:]
        number_pts = 0
        # 在固定图像检测图上进行 NMS 提取候选点（排除掉初始标签已覆盖的区域）
        points = nms(detector_pred, nms_thresh=nms_thresh, nms_size=nms_size,
                     detector_label=initial_label, mask=True)

        # ===== 新增：利用血管 Mask 过滤候选点 =====
        # 仅保留落在血管上的点作为候选，抑制背景噪声
        if vessel_mask is not None:
            filtered_points = []
            for s, k in enumerate(points):
                if len(k) == 0:
                    filtered_points.append(k)
                    continue
                    
                # 获取点坐标 [N, 2] (x, y)
                x_coords = k[:, 0].long()
                y_coords = k[:, 1].long()
                
                # 检查 mask 值 (注意 mask 形状是 [1, H, W])
                # 确保坐标在范围内
                valid_indices = (x_coords >= 0) & (x_coords < w) & (y_coords >= 0) & (y_coords < h)
                x_coords = x_coords[valid_indices]
                y_coords = y_coords[valid_indices]
                current_k = k[valid_indices]
                
                # 采样 mask 值 (假设 mask > 0.05 即为血管)
                mask_vals = vessel_mask[s, 0, y_coords, x_coords]
                is_vessel = mask_vals > 0.05
                
                filtered_points.append(current_k[is_vessel])
            points = filtered_points

        # 1. 几何校验：跨模态重现性
        points, affine_points = mapping_points(grid_inverse, points, h, w)
        geo_points, affine_geo_points = geometric_filter(affine_detector_pred, points, affine_points,
                                                         geometric_thresh=geometric_thresh)

        # 2. 内容校验：特征唯一性 (Lowe's Ratio Test)
        content_points, affine_contend_points = content_filter(descriptor_pred, affine_descriptor_pred, geo_points,
                                                               affine_geo_points, content_thresh=content_thresh,
                                                               scale=scale)
        
        enhanced_label_pts = []
        for step in range(len(content_points)):
            # 获取原始种子点位置
            positions = torch.where(label_point_positions[step, 0] == 1)
            if len(positions) == 2:
                positions = torch.stack((positions[1], positions[0]), dim=-1)
            else:
                positions = positions[0]

            # 更新 Value Map 并获取当前轮次新增的可靠点
            final_points = update_value_map(value_map[step], content_points[step], config)

            # 构造融合了种子点和新增点的临时标签
            temp_label = torch.zeros([h, w]).to(detector_pred.device)
            temp_label[final_points[:, 1], final_points[:, 0]] = 0.5
            temp_label[positions[:, 1], positions[:, 0]] = 1

            # 再次 NMS 确保标签点不重叠
            enhanced_kps = nms(temp_label.unsqueeze(0).unsqueeze(0), 0.1, 10)[0]
            if len(enhanced_kps) < len(positions):
                enhanced_kps = positions
            
            number_pts += (len(enhanced_kps) - len(positions))

            temp_label[:] = 0
            temp_label[enhanced_kps[:, 1], enhanced_kps[:, 0]] = 1
            enhanced_label_pts.append(temp_label.unsqueeze(0).unsqueeze(0))

            # 生成高斯平滑后的软标签，用于 Dice Loss 训练
            temp_label = F.conv2d(temp_label.unsqueeze(0).unsqueeze(0), kernel, stride=1,
                                  padding=(kernel.shape[-1] - 1) // 2)
            temp_label[temp_label > 1] = 1

            if enhanced_label is None:
                enhanced_label = temp_label
            else:
                enhanced_label = torch.cat((enhanced_label, temp_label))

    enhanced_label_pts = torch.cat(enhanced_label_pts)
    
    # 计算几何一致性损失 (l_geo)：将运动图像的检测图投影回固定图像坐标系
    affine_pred_inverse = F.grid_sample(affine_detector_pred, grid_inverse, align_corners=True)

    # l_clf: 拟合动态演化后的标签 Y_t
    loss1 = loss_cal(detector_pred, enhanced_label)
    # l_geo: 强制跨模态检测图在空间上对齐
    loss2 = loss_cal(detector_pred, affine_pred_inverse)

    loss = loss1 + loss2

    return loss, number_pts, value_map, enhanced_label_pts, enhanced_label
