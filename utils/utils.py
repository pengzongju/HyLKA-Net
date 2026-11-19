import numpy as np
import os
from skimage import metrics
import torch
from pathlib import Path
import matplotlib.pyplot as plt
import logging
from option import args
from einops import rearrange
import xlwt
import torch.nn.functional as F
import torch
import cv2
from scipy.ndimage import prewitt, gaussian_filter
from scipy.fft import fft
import torch.nn as nn


class ExcelFile():
    def __init__(self):
        self.xlsx_file = xlwt.Workbook()
        self.worksheet = self.xlsx_file.add_sheet(r'sheet1', cell_overwrite_ok=True)
        self.worksheet.write(0, 0, 'Datasets')
        self.worksheet.write(0, 1, 'Scenes')
        self.worksheet.write(0, 2, 'PSNR')
        self.worksheet.write(0, 3, 'SSIM')
        self.worksheet.write(0, 4, 'MAE')
        self.worksheet.write(0, 5, 'LPIPS')
        self.worksheet.write(0, 6, 'LF-IQA')
        self.worksheet.col(6).width = 256 * 12
        self.worksheet.col(0).width = 256 * 16
        self.worksheet.col(1).width = 256 * 22
        self.worksheet.col(2).width = 256 * 10
        self.worksheet.col(3).width = 256 * 10
        self.worksheet.col(4).width = 256 * 10
        self.worksheet.col(5).width = 256 * 10
        self.sum = 1

    def write_sheet(self, test_name, LF_name, psnr_iter_test, ssim_iter_test, mae_iter_test,lpips_iter_test,lfiqa_iter_test):#test
        ''' Save PSNR & SSIM $ MAE

        Args:
            test:
        '''
        for i in range(len(psnr_iter_test)):
            self.add_sheet(test_name, LF_name[i], psnr_iter_test[i], ssim_iter_test[i], mae_iter_test[i],lpips_iter_test[i],lfiqa_iter_test[i])

        psnr_epoch_test = float(np.array(psnr_iter_test).mean())
        ssim_epoch_test = float(np.array(ssim_iter_test).mean())
        mae_epoch_test = float(np.array(mae_iter_test).mean())
        lpips_epoch_test = float(np.array(lpips_iter_test).mean())
        lfiqa_epoch_test = float(np.array(lfiqa_iter_test).mean())
        self.add_sheet(test_name, 'average', psnr_epoch_test, ssim_epoch_test,mae_epoch_test,lpips_epoch_test,lfiqa_epoch_test)
        self.sum = self.sum + 1

    def add_sheet(self, test_name, LF_name, psnr_iter_test, ssim_iter_test,mae_iter_test,lpips_iter_test,lfiqa_iter_test):
        ''' Save PSNR & SSIM '''
        self.worksheet.write(self.sum, 0, test_name)
        self.worksheet.write(self.sum, 1, LF_name)
        self.worksheet.write(self.sum, 2, '%.6f' % psnr_iter_test)
        self.worksheet.write(self.sum, 3, '%.6f' % ssim_iter_test)
        self.worksheet.write(self.sum, 4, '%.6f' % mae_iter_test)
        self.worksheet.write(self.sum, 5, '%.6f' % lpips_iter_test)
        self.worksheet.write(self.sum, 6, '%.6f' % lfiqa_iter_test)  # 新增

        self.sum = self.sum + 1


def get_logger(log_dir, args):
    '''LOG '''
    logger = logging.getLogger(args.model_name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model_name))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def create_dir(args):
    log_dir = Path(args.path_log)
    log_dir.mkdir(exist_ok=True)
    if args.task == 'SR':
        task_path = 'SR_' + str(args.angRes_in) + 'x' + str(args.angRes_in) + '_' + str(args.scale_factor) + 'x'
    
    log_dir = log_dir.joinpath(task_path)
    log_dir.mkdir(exist_ok=True)
    log_dir = log_dir.joinpath(args.data_name)
    log_dir.mkdir(exist_ok=True)
    log_dir = log_dir.joinpath(args.model_name)
    log_dir.mkdir(exist_ok=True)

    checkpoints_dir = log_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)

    results_dir = log_dir.joinpath('results/')
    results_dir.mkdir(exist_ok=True)

    return log_dir, checkpoints_dir, results_dir


class Logger():
    def __init__(self, log_dir, args):
        self.logger = get_logger(log_dir, args)

    def log_string(self, str):
        if args.local_rank <= 0:
            self.logger.info(str)
            print(str)
def mae(prediction, target):
    prediction = torch.tensor(prediction)

    target = torch.tensor(target)
    absolute_errors = torch.abs(prediction - target)
    #return absolute_errors
    return torch.mean(absolute_errors).item()

import lpips

def Lpips(prediction, target):
    loss_in = lpips.LPIPS(net='alex')
    prediction = torch.tensor(prediction)
    target = torch.tensor(target)
    LPIPS = loss_in(prediction,target)
    return LPIPS.item()


def get_global_spatial_quality(ref_view, dist_view):
    """
    计算单个视图的全局空间质量 (qc)
    基于论文 III-A 节: "Global Spatial Quality Estimation" [cite: 167]
    """
    # 论文使用 FAST 角点检测 [cite: 196]
    # 我们期望输入是 0-1 的浮点数，将其转换为 0-255 uint8 以便 CV2 处理
    ref_view_u8 = (np.clip(ref_view, 0,  255)).astype(np.uint8)
    dist_view_u8 = (np.clip(dist_view, 0,  255)).astype(np.uint8)

    # 1. 初始化 FAST 检测器
    # 论文设置 N=12 [cite: 190]，这是 FAST-9_16 的标准 (OpenCV 默认)
    fast = cv2.FastFeatureDetector_create(threshold=10, nonmaxSuppression=True)

    # 2. 检测角点并创建结构图 S_r 和 S_d [cite: 171, 192]
    kp_ref = fast.detect(ref_view_u8, None)
    kp_dist = fast.detect(dist_view_u8, None)

    s_r = np.zeros(ref_view_u8.shape, dtype=bool)
    s_d = np.zeros(dist_view_u8.shape, dtype=bool)

    for kp in kp_ref:
        y, x = int(kp.pt[1]), int(kp.pt[0])
        s_r[y, x] = True

    for kp in kp_dist:
        y, x = int(kp.pt[1]), int(kp.pt[0])
        s_d[y, x] = True

    # 3. 计算 N_r (参考图中的角点) 和 N_o (匹配的角点) [cite: 204]
    n_r = np.sum(s_r)
    n_o = np.sum(np.logical_and(s_r, s_d))  # S_o (重叠) [cite: 201]

    # 4. 计算 qc (Eq. 6) [cite: 211, 231]
    qc = n_o / (n_r + 1.0)  # +1 是为了避免分母为零 [cite: 231]

    return qc


def get_local_spatial_quality(ref_view, dist_view):
    """
    计算单个视图的局部空间质量 (qe)
    基于论文 III-B 节: "Local Spatial Quality Estimation" [cite: 232]
    """
    # 1. 在参考视图上检测近边缘区域 (Eq. 7-10) [cite: 237-250]
    # 使用 Prewitt 算子 [cite: 237]
    grad_x = prewitt(ref_view, axis=1) / 3.0  # Gx [cite: 241]
    grad_y = prewitt(ref_view, axis=0) / 3.0  # Gy [cite: 244]

    grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)  # G (Eq. 8) [cite: 240]

    # 用高斯滤波器模糊梯度图，得到近边缘图 E (Eq. 10) [cite: 248]
    # 论文未指定 sigma，我们选一个合理的值
    sigma_e = 7.0
    e_map = gaussian_filter(grad_mag, sigma=sigma_e)  # E=(e_ij) [cite: 248]

    # 2. 计算加权 MSE (Near-Edge MSE) (Eq. 12) [cite: 255]
    error_sq = (ref_view - dist_view) ** 2

    numerator = np.sum(error_sq * e_map)
    denominator = np.sum(e_map)

    # 避免除以零
    if denominator == 0:
        qe = np.mean(error_sq)  # 退化为普通 MSE
    else:
        qe = numerator / denominator

    return qe


def get_angular_quality(qe_sequence, V,L=255.0):
    """
    计算整个光场的角度质量 (qa)
    基于论文 III-C 节: "Angular Quality Estimation" [cite: 259]

    qe_sequence: 一个包含 V 个视图的 qe 分数的一维数组
    """
    # 论文中的常数
    c = 1000.0  # (Eq. 18) [cite: 432]
    L = 255.0  # 像素动态范围 (我们用 0-1)
    epsilon = (0.01 * L) ** 2  # (Eq. 18) [cite: 428]
    threshold = 0.15  # (Eq. 18) [cite: 429]

    # 1. 计算 qe 序列的幅度谱 A (Eq. 15) [cite: 400]
    A = np.abs(fft(qe_sequence))
    A0 = A[0]  # DC 分量

    # 2. 找到主频率峰 k* (Eq. 16) [cite: 406]
    # 我们只在频谱的前半部分 (不含 DC) 寻找最大峰
    if V < 2:
        return c / (np.mean(qe_sequence) + epsilon)  # 视图太少，无法分析

    spectrum_half = A[1:V // 2 + 1]
    if len(spectrum_half) == 0:
        k_star = 1  # 默认
        A_k_star = 0
    else:
        A_k_star = np.max(spectrum_half)
        k_star = np.argmax(spectrum_half) + 1  # +1 是因为我们跳过了 DC

    # 3. 检查峰值是否显著 (Eq. 18) [cite: 418, 429]
    if A0 == 0 or (A_k_star / A0) >= threshold:
        # 情况 1: 存在显著的角度失真 (如重建)
        qa = k_star  # (Eq. 17) [cite: 411]
    else:
        # 情况 2: 没有显著角度失真 (如压缩) [cite: 414, 417]
        mean_qe = np.mean(qe_sequence)
        qa = c / (mean_qe + epsilon)

    return qa


def calculate_lf_iqa(ref_lf, dist_lf, angRes,L=255.0):
    """
    计算最终的 LF-IQA 指标 (Q)

    ref_lf, dist_lf: 4D 光场 (U, V, H, W)，0-1 浮点数
    angRes: 角度分辨率 (U 或 V)
    """
    U, V, H, W = ref_lf.shape
    total_views = U * V

    qc_list = []
    qe_list = []

    # 1. 逐视图计算 qc 和 qe
    for u in range(U):
        for v in range(V):
            ref_view = ref_lf[u, v, :, :]
            dist_view = dist_lf[u, v, :, :]

            # 计算 qc
            qc_list.append(get_global_spatial_quality(ref_view, dist_view))

            # 计算 qe
            qe_list.append(get_local_spatial_quality(ref_view, dist_view))

    qe_sequence = np.array(qe_list)
    qc_sequence = np.array(qc_list)

    # 2. 计算角度质量 qa (使用 qe 序列)
    qa = get_angular_quality(qe_sequence, total_views,L=L)

    # 3. 整合所有组件 (Eq. 19) [cite: 435]
    L = 255.0
    epsilon = (0.01 * L) ** 2  # (Eq. 19) [cite: 440]

    # 确保 qe_sequence > 0
    qe_sequence_sqrt = np.sqrt(np.maximum(qe_sequence, 1e-10))

    term_inside_sum = qc_sequence / (qe_sequence_sqrt + epsilon)
    mean_term = np.mean(term_inside_sum)

    Q = np.log(np.maximum(mean_term * qa, 1e-10))

    return Q

def cal_metrics(args, label, out,):

    if len(label.size()) == 4:
        label = rearrange(label, 'b c (a1 h) (a2 w) -> b c a1 h a2 w', a1=args.angRes_in, a2=args.angRes_in)
        out = rearrange(out, 'b c (a1 h) (a2 w) -> b c a1 h a2 w', a1=args.angRes_in, a2=args.angRes_in)

    if len(label.size()) == 5:
        label = label.permute((0, 1, 3, 2, 4)).unsqueeze(0)
        out = out.permute((0, 1, 3, 2, 4)).unsqueeze(0)

    B, C, U, h, V, w = label.size()
    label_y = label[:, 0, :, :, :, :].data.cpu()
    out_y = out[:, 0, :, :, :, :].data.cpu()
    # --- [0, 255] 范围的 Y 通道 (用于 LFIQA, MG) ---
    label_y_255 = (label_y * 255.0)
    out_y_255 = (out_y * 255.0)

    PSNR = np.zeros(shape=(B, U, V), dtype='float32')
    SSIM = np.zeros(shape=(B, U, V), dtype='float32')
    #
    MAE = np.zeros(shape=(B, U, V), dtype='float32')
    LPIPS = np.zeros(shape=(B, U, V), dtype='float32')
    label_y_4d = rearrange(label_y.numpy().squeeze(0), 'u h v w -> u v h w')
    out_y_4d = rearrange(out_y.numpy().squeeze(0), 'u h v w -> u v h w')

    for b in range(B):
        for u in range(U):
            for v in range(V):
                ref_view = label_y[b, u, :, v, :].numpy()
                dist_view = out_y[b, u, :, v, :].numpy()
                PSNR[b, u, v] = metrics.peak_signal_noise_ratio(label_y[b, u, :, v, :].numpy(), out_y[b, u, :, v, :].numpy(),data_range=1.0)
                #MAE
                MAE[b,u,v] = mae(label_y[b, u, :, v, :].numpy(), out_y[b, u, :, v, :].numpy())
                #LPIPS
                # LPIPS[b,u,v] = Lpips(label_y[b, u, :, v, :].numpy(), out_y[b, u, :, v, :].numpy())

                if args.task == 'RE':
                    SSIM[b, u, v] = metrics.structural_similarity(label_y[b, u, :, v, :].numpy(),
                                                                  out_y[b, u, :, v, :].numpy(),
                                                                  gaussian_weights=True,
                                                                  sigma=1.5, use_sample_covariance=False)
                else:
                    SSIM[b, u, v] = metrics.structural_similarity(label_y[b, u, :, v, :].numpy(),
                                                                  out_y[b, u, :, v, :].numpy(),
                                                                  gaussian_weights=True, data_range=2.001)#window no-data_range 2.001
                pass

    if args.task=='RE':
        for u in range(0, args.angRes_out, (args.angRes_out - 1) // (args.angRes_in - 1)):
            for v in range(0, args.angRes_out, (args.angRes_out - 1) // (args.angRes_in - 1)):
                PSNR[:, u, v] = 0
                SSIM[:, u, v] = 0
                MAE[:, u, v] = 0
                LPIPS[:,u,v] = 0

    PSNR_mean = PSNR.sum() / np.sum(PSNR > 0)
    SSIM_mean = SSIM.sum() / np.sum(SSIM > 0)
    #MAE
    MAE_mean = MAE.sum()/np.sum(PSNR > 0)
    #LPIPS
    LPIPS_mean = LPIPS.sum()/np.sum(LPIPS>0)
    # --- 2. LF-IQA 计算 (整个光场, [0, 255] 尺度) ---
    # B=1, squeeze(0)
    label_y_4d_255 = rearrange(label_y_255.numpy().squeeze(0), 'u h v w -> u v h w')
    out_y_4d_255 = rearrange(out_y_255.numpy().squeeze(0), 'u h v w -> u v h w')
    angRes = U
    # 关键：传递 [0, 255] 的数据和 L=255.0
    LFIQA_mean = calculate_lf_iqa(label_y_4d_255, out_y_4d_255, angRes, L=255.0)

    return PSNR_mean, SSIM_mean, MAE_mean, LPIPS_mean,LFIQA_mean


def ImageExtend(Im, bdr):
    [_, _, h, w] = Im.size()
    Im_lr = torch.flip(Im, dims=[-1])
    Im_ud = torch.flip(Im, dims=[-2])
    Im_diag = torch.flip(Im, dims=[-1, -2])

    Im_up = torch.cat((Im_diag, Im_ud, Im_diag), dim=-1)
    Im_mid = torch.cat((Im_lr, Im, Im_lr), dim=-1)
    Im_down = torch.cat((Im_diag, Im_ud, Im_diag), dim=-1)
    Im_Ext = torch.cat((Im_up, Im_mid, Im_down), dim=-2)
    Im_out = Im_Ext[:, :, h - bdr[0]: 2 * h + bdr[1], w - bdr[2]: 2 * w + bdr[3]]

    return Im_out


def LFdivide(data, angRes, patch_size, stride):
    data = rearrange(data, '(a1 h) (a2 w) -> (a1 a2) 1 h w', a1=angRes, a2=angRes)
    [_, _, h0, w0] = data.size()

    bdr = (patch_size - stride) // 2
    numU = (h0 + bdr * 2 - 1) // stride
    numV = (w0 + bdr * 2 - 1) // stride
    data_pad = ImageExtend(data, [bdr, bdr+stride-1, bdr, bdr+stride-1])
    # pad = torch.nn.ReflectionPad2d(padding=(bdr, bdr+stride-1, bdr, bdr+stride-1))
    # data_pad = pad(data)
    subLF = F.unfold(data_pad, kernel_size=patch_size, stride=stride)
    subLF = rearrange(subLF, '(a1 a2) (h w) (n1 n2) -> n1 n2 (a1 h) (a2 w)',
                      a1=angRes, a2=angRes, h=patch_size, w=patch_size, n1=numU, n2=numV)

    return subLF


def LFintegrate(subLF, angRes, pz, stride, h, w):
    if subLF.dim() == 4:
        subLF = rearrange(subLF, 'n1 n2 (a1 h) (a2 w) -> n1 n2 a1 a2 h w', a1=angRes, a2=angRes)
        pass
    bdr = (pz - stride) // 2
    outLF = subLF[:, :, :, :, bdr:bdr+stride, bdr:bdr+stride]
    outLF = rearrange(outLF, 'n1 n2 a1 a2 h w -> a1 a2 (n1 h) (n2 w)')
    outLF = outLF[:, :, 0:h, 0:w]

    return outLF


def rgb2ycbcr(x):
    y = np.zeros(x.shape, dtype='double')
    y[:,:,0] =  65.481 * x[:, :, 0] + 128.553 * x[:, :, 1] +  24.966 * x[:, :, 2] +  16.0
    y[:,:,1] = -37.797 * x[:, :, 0] -  74.203 * x[:, :, 1] + 112.000 * x[:, :, 2] + 128.0
    y[:,:,2] = 112.000 * x[:, :, 0] -  93.786 * x[:, :, 1] -  18.214 * x[:, :, 2] + 128.0

    y = y / 255.0
    return y


def ycbcr2rgb(x):
    mat = np.array(
        [[65.481, 128.553, 24.966],
         [-37.797, -74.203, 112.0],
         [112.0, -93.786, -18.214]])
    mat_inv = np.linalg.inv(mat)
    offset = np.matmul(mat_inv, np.array([16, 128, 128]))
    mat_inv = mat_inv * 255

    y = np.zeros(x.shape, dtype='double')
    y[:,:,0] =  mat_inv[0,0] * x[:, :, 0] + mat_inv[0,1] * x[:, :, 1] + mat_inv[0,2] * x[:, :, 2] - offset[0]
    y[:,:,1] =  mat_inv[1,0] * x[:, :, 0] + mat_inv[1,1] * x[:, :, 1] + mat_inv[1,2] * x[:, :, 2] - offset[1]
    y[:,:,2] =  mat_inv[2,0] * x[:, :, 0] + mat_inv[2,1] * x[:, :, 1] + mat_inv[2,2] * x[:, :, 2] - offset[2]
    return y

#
# ------------------------------------------------------------------------
# TTA (几何组装策略) 辅助函数
# 请将以下所有函数添加到 utils.py 的末尾
# ------------------------------------------------------------------------
#

# --- 1. TTA 增强 (Augmentation) - 作用于 2D (H, W) Tensor ---

def augment_t0_2d(img_tensor):
    '''T0: 原始'''
    return img_tensor

def augment_t1_2d(img_tensor):
    '''T1: 顺时针旋转 90 度'''
    return torch.rot90(img_tensor, k=1, dims=[0, 1])

def augment_t2_2d(img_tensor):
    '''T2: 顺时针旋转 180 度'''
    return torch.rot90(img_tensor, k=2, dims=[0, 1])

def augment_t3_2d(img_tensor):
    '''T3: 顺时针旋转 270 度'''
    return torch.rot90(img_tensor, k=3, dims=[0, 1])

def augment_t4_2d(img_tensor):
    '''T4: 水平翻转 (W dim)'''
    return torch.flip(img_tensor, dims=[1])

def augment_t5_2d(img_tensor):
    '''T5: 水平翻转 + 旋转 90'''
    return torch.rot90(torch.flip(img_tensor, dims=[1]), k=1, dims=[0, 1])

def augment_t6_2d(img_tensor):
    '''T6: 水平翻转 + 旋转 180'''
    return torch.rot90(torch.flip(img_tensor, dims=[1]), k=2, dims=[0, 1])

def augment_t7_2d(img_tensor):
    '''T7: 水平翻转 + 旋转 270'''
    return torch.rot90(torch.flip(img_tensor, dims=[1]), k=3, dims=[0, 1])

# --- 2. TTA 逆变换 (Inverse) - 作用于 2D (H_sr, W_sr) Tensor ---

def inverse_augment_t0_2d(img_tensor):
    '''T0_inv: 原始'''
    return img_tensor

def inverse_augment_t1_2d(img_tensor):
    '''T1_inv: 逆时针旋转 90 (即 旋转 270)'''
    return torch.rot90(img_tensor, k=3, dims=[0, 1])

def inverse_augment_t2_2d(img_tensor):
    '''T2_inv: 旋转 180'''
    return torch.rot90(img_tensor, k=2, dims=[0, 1])

def inverse_augment_t3_2d(img_tensor):
    '''T3_inv: 逆时针旋转 270 (即 旋转 90)'''
    return torch.rot90(img_tensor, k=1, dims=[0, 1])

def inverse_augment_t4_2d(img_tensor):
    '''T4_inv: 水平翻转'''
    return torch.flip(img_tensor, dims=[1])

def inverse_augment_t5_2d(img_tensor):
    '''T5_inv: 旋转 270 -> 水平翻转'''
    return torch.flip(torch.rot90(img_tensor, k=3, dims=[0, 1]), dims=[1])

def inverse_augment_t6_2d(img_tensor):
    '''T6_inv: 旋转 180 -> 水平翻转'''
    return torch.flip(torch.rot90(img_tensor, k=2, dims=[0, 1]), dims=[1])

def inverse_augment_t7_2d(img_tensor):
    '''T7_inv: 旋转 90 -> 水平翻转'''
    return torch.flip(torch.rot90(img_tensor, k=1, dims=[0, 1]), dims=[1])

#
# ------------------------------------------------------------------------
# TTA (几何组装策略) 辅助函数添加结束
# ------------------------------------------------------------------------
#
