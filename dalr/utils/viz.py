# Dinomaly/models_mambaAD/utils/viz.py
import cv2
import numpy as np

def save_heatmap(img_bgr, score_map, out_path, alpha=0.6):
    """
    img_bgr: [H,W,3] uint8 (BGR)
    score_map: [H,W] float
    """
    s = score_map.astype(np.float32)
    s = (s - s.min()) / (s.max() - s.min() + 1e-6)
    s = (s * 255).astype(np.uint8)
    heat = cv2.applyColorMap(s, cv2.COLORMAP_JET)
    out  = cv2.addWeighted(img_bgr, 1.0 - alpha, heat, alpha, 0)
    cv2.imwrite(out_path, out)
