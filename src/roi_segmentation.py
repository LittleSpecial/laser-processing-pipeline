# src/roi_segmentation.py

import cv2
import numpy as np
from . import config # 导入config以获取边距

def segment_roi(img_background, img_final, blur_kernel, open_kernel_size, close_kernel_size):
    """
    通过计算P1和P3的差异，使用Otsu阈值分割，并生成一个轴对齐的椭圆ROI。
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

    if len(largest_contour) >= 5: # cv2.fitEllipse 需要至少5个点
        contour_area = cv2.contourArea(largest_contour)
        # 拟合椭圆并获取其属性
        fitted_ellipse = cv2.fitEllipse(largest_contour)
        ellipse_axes = fitted_ellipse[1]
        ellipse_area = np.pi * (ellipse_axes[0] / 2.0) * (ellipse_axes[1] / 2.0)

        if ellipse_area > 0:
            area_ratio = contour_area / ellipse_area
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
    center_coords = (int(fitted_ellipse[0][0]), int(fitted_ellipse[0][1]))
    axes_lengths = (int(fitted_ellipse[1][0] / 2), int(fitted_ellipse[1][1] / 2))
    
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