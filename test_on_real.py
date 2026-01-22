import torch
import os
import sys
import yaml
import cv2
import numpy as np
from tqdm import tqdm
import argparse
from torchvision import transforms
import torch.nn.functional as F

# Add local module path
sys.path.append(os.getcwd())

from model.super_retina_multimodal import SuperRetinaMultimodal
from common.common_util import nms, sample_keypoint_desc
from dataset.CF_OCTA_v2_repaired.cf_octa_v2_repaired_dataset import CFOCTADataset
from dataset.operation_pre_filtered_cffa.operation_pre_filtered_cffa_dataset import CFFADataset
from dataset.operation_pre_filtered_cfoct.operation_pre_filtered_cfoct_dataset import CFOCTDataset
from dataset.operation_pre_filtered_octfa.operation_pre_filtered_octfa_dataset import OCTFADataset
from measurement import calculate_metrics

def draw_matches(img1, kps1, img2, kps2, matches, save_path):
    """Draw matching lines between two images"""
    if torch.is_tensor(img1): img1 = (img1.cpu().numpy() * 255).astype(np.uint8)
    if torch.is_tensor(img2): img2 = (img2.cpu().numpy() * 255).astype(np.uint8)
    
    # Ensure grayscale for drawMatches
    if img1.ndim == 3:
        if img1.shape[2] == 3: img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        else: img1 = img1.squeeze()
    if img2.ndim == 3:
        if img2.shape[2] == 3: img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
        else: img2 = img2.squeeze()
        
    kp1_cv = [cv2.KeyPoint(x=float(pt[0]), y=float(pt[1]), size=10) for pt in kps1]
    kp2_cv = [cv2.KeyPoint(x=float(pt[0]), y=float(pt[1]), size=10) for pt in kps2]
    out_img = cv2.drawMatches(img1, kp1_cv, img2, kp2_cv, matches, None, 
                             flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imwrite(save_path, out_img)

def compute_checkerboard(img1, img2, n_grid=5):
    """Compute checkerboard image for visualization"""
    if img1.ndim == 3:
        if img1.shape[2] == 3: img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        else: img1 = img1.squeeze()
    if img2.ndim == 3:
        if img2.shape[2] == 3: img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
        else: img2 = img2.squeeze()
        
    h, w = img1.shape[:2]
    # Ensure images are same size
    if img2.shape[:2] != (h, w):
        img2 = cv2.resize(img2, (w, h))
        
    grid_h, grid_w = h // n_grid, w // n_grid
    checkerboard = np.zeros_like(img1)
    for i in range(n_grid):
        for j in range(n_grid):
            y_s, y_e = i * grid_h, (i + 1) * grid_h if i < n_grid - 1 else h
            x_s, x_e = j * grid_w, (j + 1) * grid_w if j < n_grid - 1 else w
            checkerboard[y_s:y_e, x_s:x_e] = img1[y_s:y_e, x_s:x_e] if (i + j) % 2 == 0 else img2[y_s:y_e, x_s:x_e]
    return checkerboard

def test_on_real():
    parser = argparse.ArgumentParser(description="SuperRetina Real Data Evaluation Script")
    parser.add_argument('-n', '--name', type=str, required=True, help='Experiment name')
    parser.add_argument('-latestcheckpoint', action='store_true', help='Use latest checkpoint')
    parser.add_argument('--mode', type=str, choices=['cffa', 'cfoct', 'octfa', 'cfocta'], default='cffa', help='Evaluation mode')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'], help='Dataset split')
    args = parser.parse_args()

    exp_name, mode = args.name, args.mode
    checkpoint_type = 'latestpoint' if args.latestcheckpoint else 'bestcheckpoint'
    checkpoint_path = f'./save/{mode}/{exp_name}/{checkpoint_type}/checkpoint.pth'
    output_root = f'./save/test_on_real/{mode}/{exp_name}'
    os.makedirs(output_root, exist_ok=True)
    
    log_path = os.path.join(output_root, 'evaluation_log.txt')
    log_file = open(log_path, 'w')

    def log_print(msg):
        print(msg)
        log_file.write(msg + '\n')
        log_file.flush()

    config_path = './config/train_multimodal.yaml'
    if not os.path.exists(config_path):
        log_print(f"Config not found at {config_path}, using default config...")
        model_config = {
            'nms_thresh': 0.01,
            'content_thresh': 0.7,
            'geometric_thresh': 3.0,
            'VALUE_MAP': {'is_value_map_save': False},
            'PKE': {'pke_start_epoch': 100},
            'MODEL': {'name': exp_name},
            'DATASET': {'batch_size': 1}
        }
    else:
        with open(config_path) as f: config = yaml.safe_load(f)
        model_config = {**config['MODEL'], **config['PKE'], **config['DATASET'], **config['VALUE_MAP']}

    # Initialize dataset
    if mode == 'cfocta':
        test_set = CFOCTADataset(root_dir='dataset/CF_OCTA_v2_repaired', split=args.split, mode='cf2octa')
    elif mode == 'cffa':
        test_set = CFFADataset(root_dir='dataset/operation_pre_filtered_cffa', split=args.split, mode='fa2cf')
    elif mode == 'cfoct':
        test_set = CFOCTDataset(root_dir='dataset/operation_pre_filtered_cfoct', split=args.split, mode='cf2oct')
    elif mode == 'octfa':
        test_set = OCTFADataset(root_dir='dataset/operation_pre_filtered_octfa', split=args.split, mode='fa2oct')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SuperRetinaMultimodal(model_config, device=device)
    
    if os.path.exists(checkpoint_path):
        log_print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['net'])
    else:
        log_print(f"WARNING: Checkpoint not found at {checkpoint_path}. Using uninitialized model.")
        
    model.eval()

    all_metrics = []
    
    # Matching parameters aligned with train_multimodal_v3.py
    nms_thresh = model_config.get('nms_thresh', 0.01)
    content_thresh = model_config.get('content_thresh', 0.7)
    
    transform_fix = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])
    
    transform_mov = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    log_print(f"Starting evaluation on {len(test_set)} samples...")

    for i in tqdm(range(len(test_set)), desc="Evaluating"):
        raw_data = test_set.get_raw_sample(i)
        
        # Consistent unpacking based on dataset return order
        if mode == 'cfocta': # (cf, octa, pts_cf, pts_octa, path_cf, path_octa)
            img_fix_raw, img_mov_raw, pts_fix_gt, pts_mov_gt, path_fix, path_mov = raw_data
        elif mode == 'cffa': # (fa, cf, pts_fa, pts_cf, path_fa, path_cf)
            img_mov_raw, img_fix_raw, pts_mov_gt, pts_fix_gt, path_mov, path_fix = raw_data
        elif mode == 'cfoct': # (cf, oct, pts_cf, pts_oct, path_cf, path_oct)
            img_fix_raw, img_mov_raw, pts_fix_gt, pts_mov_gt, path_fix, path_mov = raw_data
        elif mode == 'octfa': # (fa, oct, pts_fa, pts_oct, path_fa, path_oct)
            img_mov_raw, img_fix_raw, pts_mov_gt, pts_fix_gt, path_mov, path_fix = raw_data
        
        # Ensure RGB for transforms
        if img_fix_raw.ndim == 2: img_fix_raw_rgb = cv2.cvtColor(img_fix_raw, cv2.COLOR_GRAY2RGB)
        else: img_fix_raw_rgb = img_fix_raw
        if img_mov_raw.ndim == 2: img_mov_raw_rgb = cv2.cvtColor(img_mov_raw, cv2.COLOR_GRAY2RGB)
        else: img_mov_raw_rgb = img_mov_raw
        
        sample_id = os.path.basename(path_fix).split('.')[0]
        
        # Prepare inputs for model
        img0_tensor = transform_fix(img_fix_raw_rgb).unsqueeze(0).to(device)
        img1_tensor = transform_mov(img_mov_raw_rgb).unsqueeze(0).to(device)
        
        with torch.no_grad():
            # Forward flow aligned with train_multimodal_v3.py (no vessel masks)
            det_fix, desc_fix = model.network(img0_tensor)
            det_mov, desc_mov = model.network(img1_tensor)
            
            # Valid mask to avoid border artifacts
            valid_mask = (img1_tensor > -0.9).float()
            valid_mask = -F.max_pool2d(-valid_mask, kernel_size=5, stride=1, padding=2)
            det_mov_masked = det_mov * valid_mask
            
            # Keypoint extraction
            kps_fix = nms(det_fix, nms_thresh=nms_thresh, nms_size=5)[0]
            kps_mov = nms(det_mov_masked, nms_thresh=nms_thresh, nms_size=5)[0]
            
            # Fallback for sparse detections
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
                desc_fix_samp = sample_keypoint_desc(kps_fix[None], desc_fix, s=8)[0]
                desc_mov_samp = sample_keypoint_desc(kps_mov[None], desc_mov, s=8)[0]
                
                d1 = desc_fix_samp.permute(1, 0).cpu().numpy()
                d2 = desc_mov_samp.permute(1, 0).cpu().numpy()
                
                matches = cv2.BFMatcher().knnMatch(d1, d2, k=2)
                for m, n in matches:
                    if m.distance < content_thresh * n.distance:
                        good_matches.append(m)
            
            # Map back to original scale
            h_f, w_f = img_fix_raw.shape[:2]
            h_m, w_m = img_mov_raw.shape[:2]
            
            kps_f_orig = kps_fix.cpu().numpy() * [w_f / 512.0, h_f / 512.0]
            kps_m_orig = kps_mov.cpu().numpy() * [w_m / 512.0, h_m / 512.0]
            
            mkpts0 = np.array([kps_f_orig[m.queryIdx] for m in good_matches]) if good_matches else np.array([])
            mkpts1 = np.array([kps_m_orig[m.trainIdx] for m in good_matches]) if good_matches else np.array([])
            
            # Evaluation with measurement.py
            metrics = calculate_metrics(
                img_origin=img_fix_raw, img_result=img_mov_raw,
                mkpts0=mkpts0, mkpts1=mkpts1,
                kpts0=kps_f_orig, kpts1=kps_m_orig,
                ctrl_pts0=pts_fix_gt, ctrl_pts1=pts_mov_gt
            )
            
            all_metrics.append(metrics)
            
            log_print(f"ID: {sample_id} | SR_ME: {metrics['SR_ME']} | SR_MAE: {metrics['SR_MAE']} | "
                      f"Rep: {metrics['Rep']:.4f} | MIR: {metrics['MIR']:.4f} | "
                      f"MeanErr: {metrics['mean_error']:.2f} px")
            
            # Save results
            sample_save_dir = os.path.join(output_root, sample_id)
            os.makedirs(sample_save_dir, exist_ok=True)
            
            if len(mkpts0) >= 4:
                H_pred, _ = cv2.findHomography(mkpts1, mkpts0, cv2.RANSAC, 5.0)
                if H_pred is not None:
                    img_m_gray = cv2.cvtColor(img_mov_raw, cv2.COLOR_RGB2GRAY) if img_mov_raw.ndim == 3 else img_mov_raw
                    img_f_gray = cv2.cvtColor(img_fix_raw, cv2.COLOR_RGB2GRAY) if img_fix_raw.ndim == 3 else img_fix_raw
                    reg_img = cv2.warpPerspective(img_m_gray, H_pred, (w_f, h_f))
                    cv2.imwrite(os.path.join(sample_save_dir, 'registered.png'), reg_img)
                    cv2.imwrite(os.path.join(sample_save_dir, 'checkerboard.png'), compute_checkerboard(img_f_gray, reg_img))
            
            draw_matches(img_fix_raw, kps_f_orig, img_mov_raw, kps_m_orig, good_matches, 
                         os.path.join(sample_save_dir, 'matches.png'))

    # Summary
    log_print("\n" + "="*40 + "\nFINAL SUMMARY STATISTICS\n" + "="*40)
    summary = {}
    for key in ['SR_ME', 'SR_MAE', 'Rep', 'MIR', 'mean_error', 'max_error']:
        vals = [m[key] for m in all_metrics if m[key] != float('inf')]
        summary[key] = np.mean(vals) if vals else 0.0
        
    log_print(f"Overall SR_ME (Success Rate @5px):  {summary['SR_ME']*100:.2f}%")
    log_print(f"Overall SR_MAE (Success Rate @10px): {summary['SR_MAE']*100:.2f}%")
    log_print(f"Average Repeatability:              {summary['Rep']*100:.2f}%")
    log_print(f"Average Matching Inliers Ratio:     {summary['MIR']*100:.2f}%")
    log_print(f"Mean Registration Error:            {summary['mean_error']:.2f} px")
    log_print(f"Max Registration Error (Average):   {summary['max_error']:.2f} px")
    log_print("="*40)
    log_file.close()

if __name__ == "__main__":
    test_on_real()
