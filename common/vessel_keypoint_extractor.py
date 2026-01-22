import cv2
import numpy as np
from scipy import ndimage

def extract_vessel_keypoints(vessel_mask, min_distance=5):
    """
    从血管分割图中提取关键点（分叉点、交叉点、端点）
    
    Args:
        vessel_mask: 二值血管分割图 [H, W], 值为 0 或 255
        min_distance: 关键点之间的最小距离（像素），用于去重
    
    Returns:
        keypoint_mask: 稀疏的关键点掩码 [H, W], 只在关键点位置为1
    """
    # 确保是二值图
    binary = (vessel_mask > 127).astype(np.uint8)
    
    # 1. 骨架化：将粗血管变成单像素宽的中心线
    skeleton = cv2.ximgproc.thinning(binary * 255)
    skeleton = (skeleton > 0).astype(np.uint8)
    
    # 2. 检测分叉点和端点
    # 使用 3x3 卷积核统计每个骨架点的邻居数量
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=np.uint8)
    
    neighbor_count = cv2.filter2D(skeleton, -1, kernel)
    
    # 分叉点：有 3 个或更多邻居的骨架点
    junction_points = (neighbor_count >= 3) & (skeleton == 1)
    
    # 端点：只有 1 个邻居的骨架点（血管末端） - 已移除
    # endpoint_points = (neighbor_count == 1) & (skeleton == 1)
    
    # 仅保留分叉点
    keypoints = junction_points
    
    # 3. 非极大值抑制：去除距离过近的重复点
    keypoint_coords = np.column_stack(np.where(keypoints))
    
    if len(keypoint_coords) == 0:
        return np.zeros_like(vessel_mask, dtype=np.float32)
    
    # 简单的贪心 NMS
    selected = []
    keypoint_coords = keypoint_coords[np.argsort(-neighbor_count[keypoints])]  # 按邻居数排序，优先保留分叉点
    
    for coord in keypoint_coords:
        if len(selected) == 0:
            selected.append(coord)
        else:
            # 计算与已选点的最小距离
            distances = np.sqrt(np.sum((np.array(selected) - coord) ** 2, axis=1))
            if np.min(distances) > min_distance:
                selected.append(coord)
    
    # 4. 生成稀疏掩码
    keypoint_mask = np.zeros_like(vessel_mask, dtype=np.float32)
    if len(selected) > 0:
        selected = np.array(selected)
        keypoint_mask[selected[:, 0], selected[:, 1]] = 1.0
    
    return keypoint_mask


def extract_vessel_keypoints_fallback(vessel_mask, min_distance=5):
    """
    备用方案：如果 cv2.ximgproc 不可用，使用 skimage
    
    Args:
        vessel_mask: 二值血管分割图 [H, W]
        min_distance: 关键点之间的最小距离
    
    Returns:
        keypoint_mask: 稀疏的关键点掩码
    """
    try:
        from skimage.morphology import skeletonize
    except ImportError:
        raise ImportError("需要安装 scikit-image: pip install scikit-image")
    
    # 二值化
    binary = (vessel_mask > 127).astype(bool)
    
    # 骨架化
    skeleton = skeletonize(binary).astype(np.uint8)
    
    # 检测分叉点
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=np.uint8)
    
    neighbor_count = cv2.filter2D(skeleton, -1, kernel)
    
    junction_points = (neighbor_count >= 3) & (skeleton == 1)
    # endpoint_points = (neighbor_count == 1) & (skeleton == 1)
    
    keypoints = junction_points
    
    # NMS
    keypoint_coords = np.column_stack(np.where(keypoints))
    
    if len(keypoint_coords) == 0:
        return np.zeros_like(vessel_mask, dtype=np.float32)
    
    selected = []
    keypoint_coords = keypoint_coords[np.argsort(-neighbor_count[keypoints])]
    
    for coord in keypoint_coords:
        if len(selected) == 0:
            selected.append(coord)
        else:
            distances = np.sqrt(np.sum((np.array(selected) - coord) ** 2, axis=1))
            if np.min(distances) > min_distance:
                selected.append(coord)
    
    keypoint_mask = np.zeros_like(vessel_mask, dtype=np.float32)
    if len(selected) > 0:
        selected = np.array(selected)
        keypoint_mask[selected[:, 0], selected[:, 1]] = 1.0
    
    return keypoint_mask
