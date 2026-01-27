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
from measurement_SuperRetina import calculate_metrics, compute_auc

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

def compute_checkerboard(img1, img2, n_grid=4):
    """Compute checkerboard image for visualization"""
    if img1.ndim == 3:
        if img1.shape[2] == 3: img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        else: img1 = img1.squeeze()
    if img2.ndim == 3:
        if img2.shape[2] == 3: img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
        else: img2 = img2.squeeze()
        
    h, w = img1.shape[:2]
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

def visualize_descriptors_pca(desc_fix, desc_mov):
    """PCA visualization for descriptors (aligned with train_multimodal_v6.py)"""
    B, C, H, W = desc_fix.shape
    feat_fix = desc_fix[0].view(C, -1).permute(1, 0)
    feat_mov = desc_mov[0].view(C, -1).permute(1, 0)
    combined = torch.cat([feat_fix, feat_mov], dim=0)
    combined = F.normalize(combined, dim=-1)
    try:
        _, _, V = torch.pca_lowrank(combined, q=3, niter=2)
        pca_feat = torch.mm(combined, V)
        pca_feat_min = pca_feat.min(dim=0, keepdim=True)[0]
        pca_feat_max = pca_feat.max(dim=0, keepdim=True)[0]
        pca_feat = (pca_feat - pca_feat_min) / (pca_feat_max - pca_feat_min + 1e-8)
        pca_fix = pca_feat[:H*W].view(H, W, 3).cpu().numpy()
        pca_mov = pca_feat[H*W:].view(H, W, 3).cpu().numpy()
        return (pca_fix * 255).astype(np.uint8), (pca_mov * 255).astype(np.uint8)
    except:
        return np.zeros((H, W, 3), dtype=np.uint8), np.zeros((H, W, 3), dtype=np.uint8)

def test_on_real():
    parser = argparse.ArgumentParser(description="SuperRetina Real Data Evaluation Script (v6)")
    parser.add_argument('-n', '--name', type=str, required=True, help='Experiment name')
    parser.add_argument('-latest', '--latestcheckpoint', action='store_true', help='Use latest checkpoint')
    parser.add_argument('--mode', type=str, choices=['cffa', 'cfoct', 'octfa', 'cfocta'], default='cffa', help='Evaluation mode')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'], help='Dataset split')
    parser.add_argument('--geometric_thresh', '-g', type=float, help='RANSAC geometric threshold', default=3.0)
    parser.add_argument('--content_thresh', '-c', type=float, help='Lowe ratio threshold', default=0.7)
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

    # Load configuration
    config_path = './config/train_multimodal.yaml'
    if os.path.exists(config_path):
        with open(config_path) as f: config = yaml.safe_load(f)
        train_config = {**config['MODEL'], **config['PKE'], **config['DATASET'], **config['VALUE_MAP']}
    else:
        train_config = {'nms_thresh': 0.01, 'content_thresh': 0.7, 'geometric_thresh': 3.0}

    # Override with command line arguments
    nms_thresh = train_config.get('nms_thresh', 0.01)
    content_thresh = args.content_thresh
    geometric_thresh = args.geometric_thresh
    orig_size_ref = (2912, 2912)

    log_print(f"Eval Config | Content Thresh: {content_thresh} | Geometric Thresh: {geometric_thresh}")

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
    # v6 model needs shared_encoder=False for Dual-Path
    train_config['shared_encoder'] = False
    model = SuperRetinaMultimodal(train_config, device=device)
    
    if os.path.exists(checkpoint_path):
        log_print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['net'])
    else:
        log_print(f"WARNING: Checkpoint NOT found at {checkpoint_path}!")
        
    model.eval()

    all_results = []
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])

    log_print(f"Starting registration evaluation on {len(test_set)} samples...")

    with torch.no_grad():
        for i in tqdm(range(len(test_set)), desc="Evaluating"):
            raw_data = test_set.get_raw_sample(i)
            
            if mode == 'cfocta': # (cf, octa, pts_cf, pts_octa, path_cf, path_octa)
                img_f_raw, img_m_raw, pts_f_gt, pts_m_gt, path_f, path_m = raw_data
            elif mode == 'cffa':
                img_m_raw, img_f_raw, pts_m_gt, pts_f_gt, path_m, path_f = raw_data
            elif mode == 'cfoct':
                img_f_raw, img_m_raw, pts_f_gt, pts_m_gt, path_f, path_m = raw_data
            elif mode == 'octfa':
                img_m_raw, img_f_raw, pts_m_gt, pts_f_gt, path_m, path_f = raw_data
            
            sample_id = os.path.basename(path_f).split('.')[0]
            
            # Grayscale for model
            img_f_gray = cv2.cvtColor(img_f_raw, cv2.COLOR_RGB2GRAY) if img_f_raw.ndim == 3 else img_f_raw
            img_m_gray = cv2.cvtColor(img_m_raw, cv2.COLOR_RGB2GRAY) if img_m_raw.ndim == 3 else img_m_raw
            
            img0_t = transform(img_f_gray).unsqueeze(0).to(device)
            img1_t = transform(img_m_gray).unsqueeze(0).to(device)
            
            # v6 forward: mode='fix'/'mov'
            det_f, desc_f = model.network(img0_t, mode='fix')
            det_m, desc_m = model.network(img1_t, mode='mov')
            
            valid_mask = (img1_t > 0.05).float()
            valid_mask = -F.max_pool2d(-valid_mask, kernel_size=5, stride=1, padding=2)
            det_m = det_m * valid_mask
            
            # Keypoint extraction
            kps_f = nms(det_f, nms_thresh=nms_thresh, nms_size=5)[0]
            kps_m = nms(det_m, nms_thresh=nms_thresh, nms_size=5)[0]
            
            # Robust extraction fallback
            if len(kps_f) < 20:
                _, idx = torch.topk(det_f.view(-1), min(200, det_f.numel()))
                y = idx // det_f.shape[3]; x = idx % det_f.shape[3]
                kps_f = torch.stack([x, y], dim=1).float()
            if len(kps_m) < 20:
                _, idx = torch.topk(det_m.view(-1), min(200, det_m.numel()))
                y = idx // det_m.shape[3]; x = idx % det_m.shape[3]
                kps_m = torch.stack([x, y], dim=1).float()

            indices0, indices1 = [], []
            if len(kps_f) >= 4 and len(kps_m) >= 4:
                d_f = sample_keypoint_desc(kps_f[None], desc_f, s=8)[0].permute(1, 0).cpu().numpy()
                d_m = sample_keypoint_desc(kps_m[None], desc_m, s=8)[0].permute(1, 0).cpu().numpy()
                
                matches = cv2.BFMatcher().knnMatch(d_f, d_m, k=2)
                for m, n in matches:
                    if m.distance < content_thresh * n.distance:
                        indices0.append(m.queryIdx)
                        indices1.append(m.trainIdx)
            
            # Coordinate scaling factors
            h_f, w_f = img_f_raw.shape[:2]
            h_m, w_m = img_m_raw.shape[:2]
            
            # Scaling to Paper size 2912x2912
            sc_f = [orig_size_ref[1] / 512.0, orig_size_ref[0] / 512.0]
            sc_m = [orig_size_ref[1] / 512.0, orig_size_ref[0] / 512.0]
            
            kps_f_p = kps_f.cpu().numpy() * sc_f
            kps_m_p = kps_m.cpu().numpy() * sc_m
            
            mkpts0_p = kps_f_p[indices0] if indices0 else np.array([])
            mkpts1_p = kps_m_p[indices1] if indices1 else np.array([])
            
            # Scale GT Control Points to 2912x2912
            # pts_f_gt is at original image resolution [h_f, w_f]
            pts_f_p = pts_f_gt * [orig_size_ref[1] / w_f, orig_size_ref[0] / h_f]
            pts_m_p = pts_m_gt * [orig_size_ref[1] / w_m, orig_size_ref[0] / h_m]
            
            # Compute Metrics aligned with paper
            res = calculate_metrics(mkpts0_p, mkpts1_p, pts_f_p, pts_m_p, orig_size=orig_size_ref)
            all_results.append(res)
            
            log_print(f"ID: {sample_id} | Status: {res['status']:10} | MEE: {res['MEE']:.2f} | MAE: {res['MAE']:.2f}")

            # Visualization
            sample_dir = os.path.join(output_root, sample_id)
            os.makedirs(sample_dir, exist_ok=True)
            
            # PCA 
            pca_f, pca_m = visualize_descriptors_pca(desc_f, desc_m)
            cv2.imwrite(os.path.join(sample_dir, f'{sample_id}_pca_fix.png'), cv2.cvtColor(pca_f, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(sample_dir, f'{sample_id}_pca_mov.png'), cv2.cvtColor(pca_m, cv2.COLOR_RGB2BGR))
            
            # Matching Plot (on 512x512 scale for visualization)
            mkpts0_v = kps_f.cpu().numpy()[indices0]
            mkpts1_v = kps_m.cpu().numpy()[indices1]
            cv_matches = [cv2.DMatch(i, i, 0) for i in range(len(indices0))]
            draw_matches(img0_t, mkpts0_v, img1_t, mkpts1_v, cv_matches, os.path.join(sample_dir, f'{sample_id}_matches.png'))
            
            # Warping result
            if res['status'] != 'failed':
                # Need H from moving to fixed for warping
                # Calculate H on current view size for visualization
                H_vis, _ = cv2.findHomography(mkpts1_v, mkpts0_v, cv2.RANSAC, 3.0)
                if H_vis is not None:
                    img_f_v = (img0_t.cpu().numpy()[0,0] * 255).astype(np.uint8)
                    img_m_v = (img1_t.cpu().numpy()[0,0] * 255).astype(np.uint8)
                    warped = cv2.warpPerspective(img_m_v, H_vis, (512, 512))
                    checker = compute_checkerboard(img_f_v, warped)
                    cv2.imwrite(os.path.join(sample_dir, f'{sample_id}_warped.png'), warped)
                    cv2.imwrite(os.path.join(sample_dir, f'{sample_id}_checker.png'), checker)

    # Final Summary
    num_total = len(all_results)
    num_acc = sum(r['is_acceptable'] for r in all_results)
    num_inacc = sum(r['is_inaccurate'] for r in all_results)
    num_failed = sum(r['is_failed'] for r in all_results)
    
    # AUC calculation
    all_errors = []
    for r in all_results:
        if r['errors']: all_errors.extend(r['errors'])
    auc_score = compute_auc(all_errors)

    log_print("\n" + "="*50)
    log_print(f"EXPERIMENT: {exp_name} | MODE: {mode}")
    log_print("-" * 50)
    log_print(f"Acceptable Rate: {num_acc/num_total*100:.2f}% ({int(num_acc)}/{num_total})")
    log_print(f"Inaccurate Rate: {num_inacc/num_total*100:.2f}% ({int(num_inacc)}/{num_total})")
    log_print(f"Failed Rate:     {num_failed/num_total*100:.2f}% ({int(num_failed)}/{num_total})")
    log_print(f"AUC@5:           {auc5:.4f}")
    log_print(f"AUC@10:          {auc10:.4f}")
    log_print(f"AUC@20:          {auc20:.4f}")
    log_print(f"Avg AUC@5-20:    {avg_auc:.4f}")
    
    valid_mees = [r['MEE'] for r in all_results if r['MEE'] != float('inf')]
    valid_maes = [r['MAE'] for r in all_results if r['MAE'] != float('inf')]
    if valid_mees:
        log_print(f"Average MEE (Success Only): {np.mean(valid_mees):.2f} px")
        log_print(f"Average MAE (Success Only): {np.mean(valid_maes):.2f} px")
    log_print("="*50)
    log_file.close()

if __name__ == "__main__":
    test_on_real()
