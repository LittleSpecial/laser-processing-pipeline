# src/roi_segmentation.py

import cv2
import numpy as np
from . import config # 导入config以获取边距

WEIGHTED_CENTER_USE_THRESHOLD = 0.1   # 信号强度占比阈值，低于此值不采用强度中心
CENTER_SHIFT_MAX_RATIO = 0.8          # 差分中心可相对几何中心偏移的最大比例
PEAK_INTENSITY_FRACTION = 0.6         # 取最大强度一定比例以上的区域来估计峰值中心

def segment_roi(img_background, img_final, blur_kernel, open_kernel_size, close_kernel_size):
    """
    通过计算P1和P3的差异, 使用Otsu阈值分割, 并生成一个轴对齐的椭圆ROI。
    此版本动态计算裁剪尺寸。
    """
    diff_image = cv2.absdiff(img_background, img_final)
    blurred_diff = cv2.GaussianBlur(diff_image, blur_kernel, 0)
    _, otsu_thresh = cv2.threshold(blurred_diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    open_kernel = np.ones(open_kernel_size, np.uint8)
    close_kernel = np.ones(close_kernel_size, np.uint8)
    opened_mask = cv2.morphologyEx(otsu_thresh, cv2.MORPH_OPEN, open_kernel)
    closed_mask = cv2.morphologyEx(opened_mask, cv2.MORPH_CLOSE, close_kernel)

    contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("[跳过] 未找到任何轮廓。")
        return None, None, None

    largest_contour = max(contours, key=cv2.contourArea)

    x, y, w, h = cv2.boundingRect(largest_contour)
    rect_center = (x + w // 2, y + h // 2)

    if len(largest_contour) >= 5: # cv2.fitEllipse 需要至少5个点
        contour_area = cv2.contourArea(largest_contour)
        # 拟合椭圆并获取其属性
        fitted_ellipse = cv2.fitEllipse(largest_contour)
        ellipse_axes = fitted_ellipse[1]
        axes_lengths = (
            int(ellipse_axes[0] / 2),
            int(ellipse_axes[1] / 2)
        )
        ellipse_area = np.pi * (ellipse_axes[0] / 2.0) * (ellipse_axes[1] / 2.0)

        if ellipse_area > 0:
            area_ratio = contour_area / ellipse_area
            print(f"置信度: {area_ratio:.2f}")
            # 检查面积比率是否在config文件中定义的范围内
            if not (config.ELLIPSE_FIT_MIN_RATIO < area_ratio < config.ELLIPSE_FIT_MAX_RATIO):
                print(f"[跳过] ROI椭圆拟合置信度低。面积比率: {area_ratio:.2f} (可接受范围: {config.ELLIPSE_FIT_MIN_RATIO}-{config.ELLIPSE_FIT_MAX_RATIO})")
                return None, None, None # 关键：返回None以触发失败处理
        else:
            print("[跳过] ROI椭圆拟合失败，面积为0。")
            return None, None, None
    else:
        print("[跳过] ROI轮廓点数过少，无法拟合椭圆。")
        return None, None, None
    
    # # 计算外接矩形
    # x, y, w, h = cv2.boundingRect(largest_contour)
    # center_coords = (x + w // 2, y + h // 2)
    # axes_lengths = (w // 2, h // 2)
    # 使用拟合出的精确椭圆信息
    # 综合椭圆中心、轮廓几何质心以及差分强度加权质心来确定最终中心
    ellipse_center = (int(fitted_ellipse[0][0]), int(fitted_ellipse[0][1]))

    contour_center = rect_center
    contour_moments = cv2.moments(largest_contour)
    if contour_moments["m00"] > 0:
        contour_center = (
            int(contour_moments["m10"] / contour_moments["m00"]),
            int(contour_moments["m01"] / contour_moments["m00"])
        )
    else:
        contour_center = ellipse_center

    temp_mask = np.zeros_like(img_final, dtype=np.uint8)
    cv2.drawContours(temp_mask, [largest_contour], -1, 255, -1)

    weight_map = blurred_diff.astype(np.float32) * (temp_mask.astype(np.float32) / 255.0)
    weight_map -= weight_map.min()
    weighted_center = None
    _, _, _, max_loc = cv2.minMaxLoc(weight_map)
    signal_max = float(weight_map.max())
    global_max = float(blurred_diff.max())

    if global_max > 0 and signal_max > 0:
        ratio = signal_max / (global_max + 1e-6)
        if ratio >= WEIGHTED_CENTER_USE_THRESHOLD:
            peak_threshold = signal_max * PEAK_INTENSITY_FRACTION
            peak_mask = np.zeros_like(weight_map, dtype=np.uint8)
            peak_mask[weight_map >= peak_threshold] = 255
            if peak_mask.sum() == 0:
                peak_mask = (weight_map == signal_max).astype(np.uint8) * 255
            peak_moments = cv2.moments(peak_mask)
            if peak_moments["m00"] > 0:
                candidate = (
                    int(peak_moments["m10"] / peak_moments["m00"]),
                    int(peak_moments["m01"] / peak_moments["m00"])
                )
                if cv2.pointPolygonTest(largest_contour, candidate, False) >= 0:
                    weighted_center = candidate
    if weighted_center is None and signal_max > 0:
        fallback_candidate = (int(max_loc[0]), int(max_loc[1]))
        if cv2.pointPolygonTest(largest_contour, fallback_candidate, False) >= 0:
            weighted_center = fallback_candidate
    ratio = 0 if global_max == 0 else signal_max / max(global_max, 1e-6)
    if weighted_center is None:
        print("[中心调试] 未采用加权中心。ratio={:.3f}".format(ratio))
    else:
        print("[中心调试] 采用峰值中心: weighted={} contour={} axes={} ratio={:.3f}".format(
            weighted_center,
            contour_center,
            axes_lengths,
            ratio
        ))

    def clamp_center(cx, cy, w, h):
        return max(0, min(cx, w - 1)), max(0, min(cy, h - 1))

    if weighted_center is not None:
        dx = weighted_center[0] - contour_center[0]
        dy = weighted_center[1] - contour_center[1]
        max_dx = max(1, int(axes_lengths[0] * CENTER_SHIFT_MAX_RATIO))
        max_dy = max(1, int(axes_lengths[1] * CENTER_SHIFT_MAX_RATIO))
        ratio_clamped = min(max(ratio, 0.0), 1.0)
        shift_scale = max(0.3, 1.0 - 0.7 * ratio_clamped)
        scaled_dx = dx * shift_scale
        scaled_dy = dy * shift_scale
        limited_dx = int(np.clip(scaled_dx, -max_dx, max_dx))
        limited_dy = int(np.clip(scaled_dy, -max_dy, max_dy))
        print("[中心调试] dx={} dy={} max_dx={} max_dy={} scale={:.2f} limited_dx={} limited_dy={}".format(
            dx, dy, max_dx, max_dy, shift_scale, limited_dx, limited_dy
        ))

        candidate = (
            contour_center[0] + limited_dx,
            contour_center[1] + limited_dy
        )
        if cv2.pointPolygonTest(largest_contour, candidate, False) < 0:
            shrink_factor = 0.5
            reduced_dx, reduced_dy = limited_dx, limited_dy
            while (abs(reduced_dx) > 0 or abs(reduced_dy) > 0):
                reduced_dx = int(np.round(reduced_dx * shrink_factor))
                reduced_dy = int(np.round(reduced_dy * shrink_factor))
                candidate = (
                    contour_center[0] + reduced_dx,
                    contour_center[1] + reduced_dy
                )
                if cv2.pointPolygonTest(largest_contour, candidate, False) >= 0:
                    print("[中心调试] 候选中心调整后位于轮廓内 {}".format(candidate))
                    break
                if reduced_dx == 0 and reduced_dy == 0:
                    candidate = contour_center
                    break
            else:
                candidate = contour_center

        center_candidate = candidate
        print("[中心调试] 最终中心候选 {}".format(center_candidate))
    else:
        center_candidate = contour_center

    center_coords = clamp_center(center_candidate[0], center_candidate[1], img_final.shape[1], img_final.shape[0])

    # 创建掩码
    final_mask = np.zeros_like(img_final, dtype=np.uint8)
    cv2.ellipse(final_mask, center_coords, axes_lengths, 0, 0, 360, 255, -1)
    
    # CHANGED: 动态计算目标裁剪尺寸，并加入边距
    # padding = config.CROP_PADDING_PIXELS
    # target_size = (w + padding, h + padding)
    target_size = config.UNIFIED_CROP_SIZE
    
    return final_mask, center_coords, target_size # 恢复返回3个值


# def crop_and_center_roi(images, mask, center_coords, target_size):
#     """
#     根据给定的中心坐标和目标尺寸，从一组图像中裁剪出ROI，并将其居中。
#     (此函数恢复到接收 target_size 参数)
#     """
#     height, width = images[0].shape
#     center_x, center_y = center_coords
#     target_w, target_h = target_size

#     target_w, target_h = int(target_w), int(target_h)

#     x1 = max(0, center_x - target_w // 2)
#     y1 = max(0, center_y - target_h // 2)
#     x2 = min(width, center_x + target_w // 2)
#     y2 = min(height, center_y + target_h // 2)

#     actual_w = x2 - x1
#     actual_h = y2 - y1

#     centered_cropped_images = []
#     for img in images:
#         canvas = np.zeros((target_h, target_w), dtype=img.dtype)
#         paste_x = (target_w - actual_w) // 2
#         paste_y = (target_h - actual_h) // 2
#         canvas[paste_y : paste_y + actual_h, paste_x : paste_x + actual_w] = img[y1:y2, x1:x2]
#         centered_cropped_images.append(canvas)
    
#     canvas_mask = np.zeros((target_h, target_w), dtype=mask.dtype)
#     canvas_mask[paste_y : paste_y + actual_h, paste_x : paste_x + actual_w] = mask[y1:y2, x1:x2]
    
#     return centered_cropped_images, canvas_mask

def crop_and_center_roi(images, mask, center_coords, target_size):
    """
    (新版本) 根据给定的中心坐标和统一的目标尺寸，从图像中裁剪出ROI并居中。
    此版本不进行缩放，而是将裁剪内容粘贴到指定大小的画布上。
    """
    height, width = images[0].shape
    center_x, center_y = center_coords
    target_w, target_h = int(target_size[0]), int(target_size[1])

    # 计算以ROI为中心的裁剪框坐标
    x1 = max(0, center_x - target_w // 2)
    y1 = max(0, center_y - target_h // 2)
    x2 = min(width, center_x + target_w // 2)
    y2 = min(height, center_y + target_h // 2)

    # 实际从原图中裁剪出的宽高
    actual_w = x2 - x1
    actual_h = y2 - y1

    centered_cropped_images = []
    for img in images:
        # 1. 创建一个目标大小的黑色画布
        canvas = np.zeros((target_h, target_w), dtype=img.dtype)
        # 2. 计算粘贴位置，确保内容在画布上居中
        paste_x = (target_w - actual_w) // 2
        paste_y = (target_h - actual_h) // 2
        # 3. 将从原图中裁剪出的内容粘贴到画布中央
        canvas[paste_y : paste_y + actual_h, paste_x : paste_x + actual_w] = img[y1:y2, x1:x2]
        centered_cropped_images.append(canvas)
    
    # 对掩码也执行同样的操作
    canvas_mask = np.zeros((target_h, target_w), dtype=mask.dtype)
    paste_x_mask = (target_w - actual_w) // 2
    paste_y_mask = (target_h - actual_h) // 2
    canvas_mask[paste_y_mask : paste_y_mask + actual_h, paste_x_mask : paste_x_mask + actual_w] = mask[y1:y2, x1:x2]
    
    return centered_cropped_images, canvas_mask