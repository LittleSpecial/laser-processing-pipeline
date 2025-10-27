# src/main.py

import cv2
import argparse
from pathlib import Path
import json
import numpy as np
import argparse
from . import config
from .image_alignment import align_images
from .roi_segmentation import segment_roi, crop_and_center_roi
from .data_analysis import (
    calculate_log_ratio_heatmap,
    calculate_relative_change,
    calculate_quantitative_metrics,
    save_metrics_to_json
)

def save_image_safely(filepath: Path, image):
    # ... (此函数保持不变) ...
    try:
        success, buffer = cv2.imencode('.png', image)
        if not success:
            print(f"[保存失败] OpenCV无法将图像编码为PNG: {filepath.name}")
            return
        with open(str(filepath), 'wb') as f:
            f.write(buffer)
    except Exception as e:
        print(f"[保存失败] 写入文件时发生错误 {filepath.name}: {e}")

def main(input_dir, output_dir, filename_prefix=None, delay_time=None, manual_center_x=None, manual_center_y=None, heatmap_dir=None):
    if filename_prefix: print(f"--- 开始处理: {filename_prefix} ---")
    p1_path = input_dir / "P1_background.png"
    p2_path = input_dir / "P2_signal.png"
    p3_path = input_dir / "P3_final.png"
    if not all([p1_path.exists(), p2_path.exists(), p3_path.exists()]): return
    img_p1 = cv2.imread(str(p1_path), cv2.IMREAD_GRAYSCALE)
    img_p2 = cv2.imread(str(p2_path), cv2.IMREAD_GRAYSCALE)
    img_p3 = cv2.imread(str(p3_path), cv2.IMREAD_GRAYSCALE)
    output_dir.mkdir(parents=True, exist_ok=True)
    # config.HEATMAP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if heatmap_dir:
        heatmap_dir.mkdir(parents=True, exist_ok=True)
    base_name = f"{filename_prefix}_" if filename_prefix else ""

    p1_aligned, p2_aligned = None, None

    if manual_center_x is not None and manual_center_y is not None:
        print("--- 手动模式: 使用提供的坐标 (跳过图像配准) ---")
        # 在手动模式下，我们直接使用原始图像，以防配准失败
        p1_aligned = img_p1
        p2_aligned = img_p2
        center_coords = (manual_center_x, manual_center_y)
        target_crop_size = config.UNIFIED_CROP_SIZE
        roi_mask_original_size = np.ones_like(img_p3, dtype=np.uint8) * 255
    else:
        print("--- 自动模式 ---")
        print("步骤 2/5: 正在进行图像配准...")
        p1_aligned, _ = align_images(img_p3, img_p1, config.ORB_N_FEATURES, config.ORB_MATCH_RATIO)
        p2_aligned, _ = align_images(img_p3, img_p2, config.ORB_N_FEATURES, config.ORB_MATCH_RATIO)
        if p1_aligned is None or p2_aligned is None:
            print("错误: 图像配准失败。")
            return None # 配准失败，通知batch_process
        print("图像配准完成。")

        print("步骤 3/5: 正在分割ROI...")
        roi_mask_original_size, center_coords, target_crop_size = segment_roi(
            p1_aligned, 
            img_p3, 
            config.SEGMENTATION_BLUR_KERNEL, 
            config.MORPH_OPEN_KERNEL_SIZE,
            config.MORPH_CLOSE_KERNEL_SIZE
        )
        if roi_mask_original_size is None:
            print("错误: 自动ROI分割失败。")
            return None
    # save_image_safely(output_dir / f"{base_name}roi_mask_elliptical.png", roi_mask_original_size)
    print("ROI分割完成。")
    print("步骤 4/5: 正在裁剪、居中并缩放ROI...")
    images_to_center_crop = [p1_aligned, p2_aligned, img_p3]
    
    # CHANGED: 传递新的参数来裁剪和缩放
    cropped_centered_images, cropped_centered_mask = crop_and_center_roi(
        images_to_center_crop,
        roi_mask_original_size,
        center_coords,
        target_crop_size,
    )
    if cropped_centered_images is None: return
    p1_roi, p2_roi, p3_roi = cropped_centered_images
    # save_image_safely(output_dir / f"{base_name}P1_roi_centered.png", p1_roi)
    # save_image_safely(output_dir / f"{base_name}P2_roi_centered.png", p2_roi)
    # save_image_safely(output_dir / f"{base_name}P3_roi_centered.png", p3_roi)
    # save_image_safely(output_dir / f"{base_name}roi_mask_centered.png", cropped_centered_mask)
    print("ROI裁剪和居中完成。")
    print("步骤 5/5: 正在进行量化分析...")
    
    # CHANGED: 调用简单的热图函数 (不再需要background_image)
    _, log_ratio_heatmap_img = calculate_log_ratio_heatmap(
        p1_roi, 
        p2_roi, 
        cropped_centered_mask, 
        config.LOG_EPSILON, 
        config.LOG_RATIO_HEATMAP_COLORMAP
    )
    # save_image_safely(output_dir / f"{base_name}log_ratio_heatmap_centered.png", log_ratio_heatmap_img)
    
    # CHANGED: 调用简单的热图函数 (不再需要background_image)
    _, mean_rel_change, rel_change_heatmap_img = calculate_relative_change(
        p1_roi,
        p2_roi,
        cropped_centered_mask,
        config.REL_CHANGE_P1_THRESHOLD,
        config.REL_CHANGE_VMIN,
        config.REL_CHANGE_VMAX,
        config.REL_CHANGE_HEATMAP_COLORMAP
    )
    heatmap_filename = f"{base_name}relative_change_heatmap.png"
    # save_image_safely(config.HEATMAP_OUTPUT_DIR / heatmap_filename, rel_change_heatmap_img)
    if heatmap_dir:
        save_image_safely(heatmap_dir / heatmap_filename, rel_change_heatmap_img)
    else:
        save_image_safely(output_dir / heatmap_filename, rel_change_heatmap_img)
    
    metrics = calculate_quantitative_metrics(cropped_centered_mask, p1_roi, p2_roi, p3_roi)
    metrics['mean_relative_change'] = round(mean_rel_change, 6)
    # metrics_path = output_dir / f"{base_name}quantitative_metrics.json"
    # save_metrics_to_json(metrics, metrics_path)
    # print("量化分析完成。结果已保存。")
    # print("\n--- 量化指标摘要 ---")
    # print(json.dumps(metrics, indent=4))
    # print(f"SPECTRAL_DATA:{delay_time},{mean_rel_change}")
    # print("--- 处理完成 ---")
    metrics['delay'] = delay_time
    metrics['abs_delay_for_sort'] = abs(delay_time) if delay_time is not None else float('inf')
    
    # NEW: 将最终的metrics字典打印出来，以便batch_process捕获

    return metrics

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description="激光加工图像数据处理流水线")
#     parser.add_argument('--input_dir', type=str, required=True)
#     parser.add_argument('--output_dir', type=str, required=True)
#     parser.add_argument('--filename_prefix', type=str, required=False)
#     parser.add_argument('--delay', type=float, required=True, help='当前加工点的延时时间')
#     parser.add_argument('--manual_center_x', type=int, help='手动指定的ROI中心X坐标')
#     parser.add_argument('--manual_center_y', type=int, help='手动指定的ROI中心Y坐标')
#     parser.add_argument('--heatmap_dir', type=str, required=False, help='用于保存所有热图专用目录')
#     args = parser.parse_args()

#     heatmap_path = Path(args.heatmap_dir) if args.heatmap_dir else None
#     main(Path(args.input_dir), Path(args.output_dir), args.filename_prefix, args.delay, args.manual_center_x, args.manual_center_y, heatmap_path)