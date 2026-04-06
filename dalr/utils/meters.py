# Dinomaly/models_mambaAD/utils/meters.py
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_recall_curve
from skimage.measure import label, regionprops

def image_level_metrics(y_true, y_score):
    auroc = roc_auc_score(y_true, y_score)
    ap    = average_precision_score(y_true, y_score)
    # F1_max：按 PR 曲线扫阈值
    p, r, t = precision_recall_curve(y_true, y_score)
    f1 = (2 * p * r / (p + r + 1e-6)).max()
    return dict(auroc=auroc, ap=ap, f1=f1)

def pixel_level_metrics(gt_mask, score_map):
    """
    gt_mask: [N,H,W] in {0,1}
    score_map: [N,H,W] in [0,1] or real
    """
    y_true = gt_mask.reshape(-1).astype(np.uint8)
    y_score= score_map.reshape(-1).astype(np.float32)
    auroc = roc_auc_score(y_true, y_score)
    ap    = average_precision_score(y_true, y_score)
    p, r, t = precision_recall_curve(y_true, y_score)
    f1 = (2 * p * r / (p + r + 1e-6)).max()
    # AUPRO（近似）：逐阈值计算 region overlap 的平均比例再积分
    au_pro = _apro_region(gt_mask, score_map, thresholds=np.linspace(0.01, 0.99, 50))
    return dict(auroc=auroc, ap=ap, f1=f1, aupro=au_pro)

def _apro_region(gt_mask, score_map, thresholds):
    N, H, W = gt_mask.shape
    gt_mask = gt_mask.astype(np.uint8)
    smap = (score_map - score_map.min()) / (score_map.max() - score_map.min() + 1e-6)
    pros = []
    for th in thresholds:
        binmap = (smap >= th).astype(np.uint8)
        pro_n = []
        for n in range(N):
            lab = label(gt_mask[n], connectivity=1)
            regions = regionprops(lab)
            if not regions:
                continue
            total = 0.
            for reg in regions:
                yy, xx = zip(*reg.coords)
                reg_pred = binmap[n, np.array(yy), np.array(xx)]
                total += reg_pred.mean()
            pro_n.append(total / max(len(regions), 1))
        if pro_n:
            pros.append(np.mean(pro_n))
    if not pros:
        return 0.0
    # 归一化积分
    return float(np.trapz(pros, thresholds) / (thresholds[-1] - thresholds[0]))

def mAD(img_dict, pix_dict):
    """
    论文中的 7 指标均值：image: auroc/ap/f1; pixel: auroc/ap/f1/aupro
    """
    vals = [img_dict['auroc'], img_dict['ap'], img_dict['f1'],
            pix_dict['auroc'], pix_dict['ap'], pix_dict['f1'], pix_dict['aupro']]
    return float(np.mean(vals))
