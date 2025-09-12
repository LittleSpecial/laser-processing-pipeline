# src/data_analysis.py

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import regionprops
import json

# def generate_heatmap_image(data, mask, vmin, vmax, colormap):
#     """
#     根据给定的数据生成一个可视化的热图, ROI之外为黑色。
#     """
#     data_masked = np.copy(data).astype(np.float32)
#     data_masked[mask == 0] = 0 # 将ROI外部区域设为0

#     clipped_data = np.clip(data_masked, vmin, vmax)
#     if vmax > vmin:
#         normalized_data = (clipped_data - vmin) / (vmax - vmin)
#     else:
#         normalized_data = np.zeros_like(clipped_data)

#     cmap = plt.get_cmap(colormap)
#     heatmap_rgba = cmap(normalized_data)
    
#     if colormap == 'gray':
#         heatmap_final = (normalized_data * 255).astype(np.uint8)
#     else:
#         heatmap_final = (heatmap_rgba[:, :, :3] * 255).astype(np.uint8)
    
#     heatmap_final[mask == 0] = 0 # 确保背景是黑色
#     return heatmap_final

def generate_heatmap_image(data, vmin, vmax, colormap):
    """
    根据给定的数据和颜色范围生成一个可视化的热图。
    """
    # 移除所有与 mask 相关的操作
    data_float = data.astype(np.float32)

    # 裁剪数据到指定的颜色范围
    clipped_data = np.clip(data_float, vmin, vmax)
    
    # 归一化数据到 0-1 范围
    if vmax > vmin:
        normalized_data = (clipped_data - vmin) / (vmax - vmin)
    else:
        normalized_data = np.zeros_like(clipped_data)

    # 应用颜色映射
    cmap = plt.get_cmap(colormap)
    heatmap_rgba = cmap(normalized_data)
    
    # 转换回 8-bit 图像格式
    if colormap == 'gray':
        heatmap_final = (normalized_data * 255).astype(np.uint8)
    else:
        heatmap_final = (heatmap_rgba[:, :, :3] * 255).astype(np.uint8)
    
    return heatmap_final


def calculate_log_ratio_heatmap(p1_roi, p2_roi, mask_roi, epsilon, colormap):
    p1_float = p1_roi.astype(np.float32)
    p2_float = p2_roi.astype(np.float32)
    log_ratio_data = np.log((p2_float + epsilon) / (p1_float + epsilon))
    
    roi_pixels = log_ratio_data[mask_roi != 0]
    vmin, vmax = np.min(roi_pixels), np.max(roi_pixels) if roi_pixels.size > 0 else (0, 0)
    
    heatmap_img = generate_heatmap_image(log_ratio_data, vmin, vmax, colormap)
    return log_ratio_data, heatmap_img

def calculate_relative_change(p1_roi, p2_roi, mask_roi, p1_threshold, vmin, vmax, colormap):
    p1_float = p1_roi.astype(np.float32)
    p2_float = p2_roi.astype(np.float32)

    p1_safe = p1_float.copy()
    p1_safe[p1_safe < p1_threshold] = p1_threshold 
    
    relative_change = (p2_float - p1_float) / p1_safe 
    # relative_change[mask_roi == 0] = 0

    roi_pixels = relative_change[mask_roi != 0]
    mean_change = roi_pixels.mean() if roi_pixels.size > 0 else 0
    
    heatmap_img = generate_heatmap_image(relative_change, vmin, vmax, colormap)

    return relative_change, float(mean_change), heatmap_img

def calculate_quantitative_metrics(mask_roi, p1_roi, p2_roi, p3_roi):
    # ... (此函数保持不变) ...
    metrics = {}
    binary_mask = (mask_roi > 0).astype(np.uint8)
    props = regionprops(binary_mask)
    if not props: return metrics
    prop = props[0]
    metrics['area_pixels'] = int(prop.area)
    metrics['perimeter'] = round(prop.perimeter, 2)
    metrics['solidity'] = round(prop.solidity, 4)
    metrics['equivalent_diameter'] = round(prop.equivalent_diameter, 2)
    metrics['mean_intensity_P1'] = round(cv2.mean(p1_roi, mask=mask_roi)[0], 2)
    metrics['mean_intensity_P2'] = round(cv2.mean(p2_roi, mask=mask_roi)[0], 2)
    metrics['mean_intensity_P3'] = round(cv2.mean(p3_roi, mask=mask_roi)[0], 2)
    return metrics

def save_metrics_to_json(metrics, filepath):
    # ... (此函数保持不变) ...
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)