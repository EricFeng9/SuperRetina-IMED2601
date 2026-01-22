import configparser
import json

import numpy as np
import torch
from tqdm import tqdm

from common.eval_util import compute_auc
from predictor import Predictor
import os
import cv2
import yaml


def compute_auc_rop(s_error):
    s_error = np.array(s_error)
    limit = 25
    gs_error = np.zeros(limit + 1)
    accum_s = 0
    for i in range(1, limit + 1):
        gs_error[i] = np.sum(s_error < i) * 100 / len(s_error)

        accum_s = accum_s + gs_error[i]

    auc_s = accum_s / (limit * 100)
    # auc_p = accum_p / (limit * 100)
    # auc_a = accum_a / (limit * 100)
    # mAUC = (auc_s + auc_p + auc_a) / 3.0
    return {'mAUC': auc_s}


def getPointsFromJSON(data):
    sorted_shapes = sorted(data['shapes'], key=lambda x: int(x['label']))
    all_points = []
    for shape in sorted_shapes:
        all_points.extend(shape['points'])
    return all_points


def cal_MSE(src, dst, theta):
    points_warped = cv2.perspectiveTransform(src.reshape(-1, 1, 2), theta).reshape(-1, 2)
    # for i in range(len(dst)):
    #     print(src[i], points_warped[i], dst[i])
    loss_in1 = torch.Tensor(points_warped)
    loss_in2 = torch.Tensor(dst)
    MSE = torch.nn.MSELoss()
    loss = MSE(loss_in1, loss_in2)
    # for i in range(len(src)):
    #     print(dst[i] - points_warped[i])
    # input("===")
    return points_warped, loss


def cv_imread(file_path):
    cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
    if cv_img is not None and len(cv_img.shape) == 2:  # 单通道图像
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2BGR)  # 转换为三通道图像
    return cv_img


def checkerboard(I1, I2, n, dim=2):
    # print(I1.shape, I2.shape)
    assert I1.shape == I2.shape
    if dim == 2:
        height, width = I1.shape
    else:
        height, width, channel = I1.shape

    hi, wi = height / n, width / n
    if dim == 2:
        outshape = (int(hi * n), int(wi * n))
    else:
        outshape = (int(hi * n), int(wi * n), channel)

    out_image = np.zeros(outshape, dtype='uint8')
    for i in range(n):
        h = int(round(hi * i))
        h1 = int(round(h + hi))
        for j in range(n):
            w = int(round(wi * j))
            w1 = int(round(w + wi))

            if (i - j) % 2 == 0:
                if dim == 2:
                    out_image[h:h1, w:w1] = I1[h:h1, w:w1]
                else:
                    out_image[h:h1, w:w1, :] = I1[h:h1, w:w1, :]
            else:
                if dim == 2:
                    out_image[h:h1, w:w1] = I2[h:h1, w:w1]
                else:
                    out_image[h:h1, w:w1, :] = I2[h:h1, w:w1, :]
    return out_image


config_path = './config/test.yaml'
if os.path.exists(config_path):
    with open(config_path) as f:
        config = yaml.safe_load(f)
else:
    raise FileNotFoundError("Config File doesn't Exist")

Pred = Predictor(config)

data_path = '../data'  # Change the data_path according to your own setup
testset = 'operation_B2A'
use_matching_trick = config['PREDICT']['use_matching_trick']
# gt_dir = os.path.join(data_path, testset, 'Ground Truth')
# im_dir = os.path.join(data_path, testset, 'Images')
ROP_dir = os.path.join(data_path, testset)
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

big_num = 1e6
good_nums_rate = []
image_num = 0

failed = 0
inaccurate = 0
mae = 0
mee = 0

# category: S, P, A, corresponding to Easy, Hard, Mod in paper
auc_record = []
mse_record = []
cnt = 0
method = "CFFA_SuperRetina_gmd_pretrained"

#
# for subdir, dirs, files in tqdm(all_dirs, desc="Scanning directories"):
# # for subdir, dirs, files in os.walk(ROP_dir):
#     txt_files = [file for file in files if file.endswith('.txt')]
#     # 遍历当前目录下的所有文件
#     # print(f"==== checking '{subdir}', have {len(txt_files)} TXT files ====")

all_dirs = list(os.walk(ROP_dir))

pbar = tqdm(all_dirs, desc="Scanning directories")
for subdir, dirs, files in pbar:
    txt_files = [file for file in files if file.endswith('.txt')]

    # 显示当前文件夹和文件数量
    folder_name = os.path.basename(subdir) if subdir else "root"
    pbar.set_postfix(dir=folder_name, txt_files=len(txt_files))

    for i in range(len(txt_files)):
        for j in range(i + 1, len(txt_files)):
            dir_name = subdir.split('/')[-1].split('-')
            # dir_idx = int(subdir.split('/')[-1].split('-')[0].split('_')[0])
            # if dir_idx > -1:
            if True:
                src_path = os.path.join(subdir, txt_files[i])
                trg_path = os.path.join(subdir, txt_files[j])

                # 获取图片的路径，并动态确定图片的扩展名
                src_img_path = os.path.join(subdir, txt_files[i][:-4])
                trg_img_path = os.path.join(subdir, txt_files[j][:-4])

                # 获取同名图片文件的扩展名，支持多个常见格式
                possible_exts = ['.png', '.jpg', '.jpeg']
                src_img_path_with_ext = None
                trg_img_path_with_ext = None

                # 判断图片文件的扩展名并设置完整路径
                for ext in possible_exts:
                    if os.path.exists(src_img_path + ext):
                        src_img_path = src_img_path + ext
                        break
                for ext in possible_exts:
                    if os.path.exists(trg_img_path + ext):
                        trg_img_path = trg_img_path + ext
                        break

                src_name = src_img_path.split('/')[-1][:-4]
                trg_name = trg_img_path.split('/')[-1][:-4]

                with open(src_path, 'r', encoding='utf-8') as f:
                    src = np.loadtxt(src_path)
                with open(trg_path, 'r', encoding='utf-8') as f:
                    trg = np.loadtxt(trg_path)
                assert len(src) == len(trg)

                img_src = cv_imread(src_img_path)
                img_trg = cv_imread(trg_img_path)
                if img_src.shape != (768, 768, 3):
                    h, w = img_src.shape[:2]
                    src[:, 0] = src[:, 0] / w * 768  # x 方向缩放
                    src[:, 1] = src[:, 1] / h * 768  # y 方向缩放
                    # print("changed GT")
                    # src = src / img_src.shape[0] * 768
                    img_src = cv2.resize(img_src, (768, 768))
                if img_trg.shape != (768, 768, 3):
                    h, w = img_trg.shape[:2]
                    trg[:, 0] = trg[:, 0] / w * 768  # x 方向缩放
                    trg[:, 1] = trg[:, 1] / h * 768  # y 方向缩放
                    # trg = trg / img_trg.shape[0] * 768
                    img_trg = cv2.resize(img_trg, (768, 768))
                assert len(src) == len(trg)

                query_im_path = src_img_path
                refer_im_path = trg_img_path
                H_m1, inliers_num_rate, query_image, _ = Pred.compute_homography(query_im_path, refer_im_path,
                                                                                 resize=(768, 768))
                H_m2 = None
                if use_matching_trick:
                    if H_m1 is not None:
                        h, w = Pred.image_height, Pred.image_width
                        query_align_first = cv2.warpPerspective(query_image, H_m1, (w, h),
                                                                borderMode=cv2.BORDER_CONSTANT,
                                                                borderValue=(0))
                        query_align_first = query_align_first.astype(float)
                        query_align_first /= 255.
                        H_m2, inliers_num_rate, _, _ = Pred.compute_homography(query_align_first, refer_im_path,
                                                                               query_is_image=True, resize=(768, 768))

                good_nums_rate.append(inliers_num_rate)
                image_num += 1

                if inliers_num_rate < 1e-6:
                    # inaccurate += 1
                    failed += 1
                    avg_dist = big_num
                    auc_record.append(avg_dist)
                else:
                    raw = src
                    dst = trg
                    dst_pred = cv2.perspectiveTransform(raw.reshape(-1, 1, 2), H_m1)
                    pred, mse = cal_MSE(src, trg, H_m1)

                    if H_m2 is not None:
                        dst_pred = cv2.perspectiveTransform(dst_pred.reshape(-1, 1, 2), H_m2)
                        pred, mse = cal_MSE(pred, trg, H_m2)
                    M, mask = cv2.findHomography(src, pred)
                    # error = trg - pred
                    # rmse_lst = []
                    # for k in range(error.shape[0]):
                    #     err = np.sqrt(np.square(error[k][0]) + np.square(error[k][1]))
                    #     rmse_lst.append(err)
                    # rmse = np.average(rmse_lst)
                    dst_pred = dst_pred.squeeze()
                    dis = (dst - dst_pred) ** 2
                    dis = np.sqrt(dis[:, 0] + dis[:, 1])
                    avg_dist = dis.mean()
                    # loss_in1 = torch.Tensor(dst_pred)
                    # loss_in2 = torch.Tensor(dst)
                    # MSE = torch.nn.MSELoss()
                    # mse = MSE(loss_in1, loss_in2)

                    mae = dis.max()
                    mee = np.median(dis)
                    if mae > 50 or mee > 20:
                        inaccurate += 1
                    else:
                        mse_record.append(mse)
                    auc_record.append(avg_dist)

                    # img_src = cv_imread(src_img_path)
                    # img_trg = cv_imread(trg_img_path)
                    # if M is not None:
                    #     img_warp = cv2.warpPerspective(img_src, M, (img_src.shape[1], img_src.shape[0]))
                    #     img_checkerboard = checkerboard(img_trg, img_warp, 4, 3)
                    #     chessboard_path = os.path.join('.', "output", method, 'checkerboard',
                    #                                    f"{dir_name}" + "_" + src_name + "_" + trg_name + '_checkerboard.jpg')
                    #     os.makedirs(os.path.dirname(chessboard_path), exist_ok=True)
                    #     cv2.imwrite(chessboard_path, img_checkerboard)
                    # img_trg_path = os.path.join('.', "output", method,
                    #                             f"{dir_name}" + "_" + src_name + "_" + trg_name + '_trg.jpg')
                    # img_warp_path = os.path.join('.', "output", method,
                    #                              f"{dir_name}" + "_" + src_name + "_" + trg_name + '_warp.jpg')
                    # cv2.imwrite(img_trg_path, img_trg)
                    # cv2.imwrite(img_warp_path, img_warp)
                    # # matrix_path = os.path.join('.', "output", method,
                    # #                            f"{idx:03d}" + "_" + src_name + "_" + trg_name + '_matrix.json')
                    # # save_transformation_matrix(M, matrix_path)

                cnt += 1

print('-' * 40)
print('pth:' + config['PREDICT']['model_save_path'])
print('dataset:' + testset)
print(f"Failed:{'%.2f' % (100 * failed / image_num)}%, Inaccurate:{'%.2f' % (100 * inaccurate / image_num)}%, "
      f"Acceptable:{'%.2f' % (100 * (image_num - inaccurate - failed) / image_num)}%")

print('-' * 40)

auc = compute_auc_rop(auc_record)
print('failed %d' % failed)
print('mAUC: %.3f, cnt %d' % (auc['mAUC'], len(auc_record)))
print(np.mean(np.array(mse_record)))
print('totally %d image pairs, inaccurate: %d, mAUC: %.3f' % (cnt, inaccurate, auc['mAUC']))