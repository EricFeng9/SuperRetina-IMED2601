import numpy as np
import cv2

def spatial_binning(pts0, pts1, img_size=(2912, 2912), grid_size=4, top_n=20):
    """
    Spatial Binning to improve RANSAC robustness by ensuring uniform distribution of points.
    """
    h, w = img_size
    cell_h = h / grid_size
    cell_w = w / grid_size
    
    selected_indices = []
    grid = [[] for _ in range(grid_size * grid_size)]
    
    for i, pt in enumerate(pts0):
        gx = min(int(pt[0] / cell_w), grid_size - 1)
        gy = min(int(pt[1] / cell_h), grid_size - 1)
        grid[gy * grid_size + gx].append(i)
        
    for cell_indices in grid:
        if len(cell_indices) == 0:
            continue
        selected_indices.extend(cell_indices[:top_n])
        
    return np.array(selected_indices)

def calculate_metrics(mkpts0, mkpts1, ctrl_pts0, ctrl_pts1, orig_size=(2912, 2912)):
    """
    Performance Metrics aligned with the SuperRetina paper.
    
    Args:
    - mkpts0: Matches in fixed image (already scaled to orig_size) [N, 2]
    - mkpts1: Matches in moving image (already scaled to orig_size) [N, 2]
    - ctrl_pts0: GT control points in fixed image (already scaled to orig_size) [M, 2]
    - ctrl_pts1: GT control points in moving image (already scaled to orig_size) [M, 2]
    - orig_size: The size at which metrics are computed, default (2912, 2912)
    
    Returns:
    - results: Dictionary containing MEE, MAE, Status, and binary rates.
    """
    
    # Paper Thresholds (on 2912x2912)
    THR_MEE = 20.0
    THR_MAE = 50.0
    RANSAC_THR = 5.0 # Typical RANSAC tolerance
    
    results = {
        'MEE': float('inf'),
        'MAE': float('inf'),
        'status': 'failed', # 'failed', 'inaccurate', 'acceptable'
        'is_failed': 1.0,
        'is_inaccurate': 0.0,
        'is_acceptable': 0.0,
        'errors': [] # L2 distances for each control point
    }
    
    # 1. Failed if number of matches is less than 4
    if len(mkpts0) < 4:
        results['status'] = 'failed'
        return results

    # 2. Estimate Homography H (from moving to fixed)
    # Using Spatial Binning for robustness as per paper's spirit
    bin_indices = spatial_binning(mkpts1, mkpts0, img_size=orig_size)
    if len(bin_indices) >= 4:
        H_pred, _ = cv2.findHomography(mkpts1[bin_indices], mkpts0[bin_indices], cv2.RANSAC, RANSAC_THR)
    else:
        H_pred, _ = cv2.findHomography(mkpts1, mkpts0, cv2.RANSAC, RANSAC_THR)

    if H_pred is None:
        results['status'] = 'failed'
        return results

    # 3. Compute Reprojection Error on original size
    ctrl_pts0 = np.array(ctrl_pts0).reshape(-1, 2)
    ctrl_pts1 = np.array(ctrl_pts1).reshape(-1, 2)
    
    # Warp moving control points to fixed image space
    pts1_h = np.concatenate([ctrl_pts1, np.ones((len(ctrl_pts1), 1))], axis=1)
    pts1_warped_h = (H_pred @ pts1_h.T).T
    pts1_warped = pts1_warped_h[:, :2] / (pts1_warped_h[:, 2:] + 1e-7)
    
    # Compute L2 distances
    errors = np.linalg.norm(pts1_warped - ctrl_pts0, axis=1)
    mee = np.median(errors) # Median Error
    mae = np.max(errors)    # Maximum Error
    
    results['MEE'] = float(mee)
    results['MAE'] = float(mae)
    results['errors'] = errors.tolist()
    
    # 4. Classification: Acceptable if MEE < 20 and MAE < 50
    if mee < THR_MEE and mae < THR_MAE:
        results['status'] = 'acceptable'
        results['is_acceptable'] = 1.0
        results['is_failed'] = 0.0
        results['is_inaccurate'] = 0.0
    else:
        results['status'] = 'inaccurate'
        results['is_inaccurate'] = 1.0
        results['is_failed'] = 0.0
        results['is_acceptable'] = 0.0
        
    return results

def compute_auc(all_errors, max_threshold=50, num_steps=1000):
    """
    Computes Area Under Curve (AUC) for acceptance rates w.r.t. decision threshold.
    Higher is better.
    """
    if len(all_errors) == 0:
        return 0.0
    
    errors = np.array(all_errors)
    thresholds = np.linspace(0, max_threshold, num_steps)
    # Recall (Accuracy) at each threshold
    acc_rates = [np.mean(errors <= t) for t in thresholds]
    
    # Area under the curve normalized by max_threshold
    auc = np.trapz(acc_rates, thresholds) / max_threshold
    return float(auc)

if __name__ == "__main__":
    # Test script
    mkpts0 = np.random.rand(10, 2) * 2912
    mkpts1 = mkpts0 + 5.0
    ctrl_0 = np.random.rand(6, 2) * 2912
    ctrl_1 = ctrl_0 - 5.0
    
    res = calculate_metrics(mkpts0, mkpts1, ctrl_0, ctrl_1)
    print(res)
