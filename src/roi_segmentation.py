import cv2
import numpy as np

from . import config


def _apply_contrast_enhancement(image: np.ndarray) -> np.ndarray:
    """Improve contrast so weak laser responses are more separable."""
    if not config.SEGMENTATION_USE_CLAHE:
        return image
    clahe = cv2.createCLAHE(
        clipLimit=config.CLAHE_CLIP_LIMIT,
        tileGridSize=config.CLAHE_TILE_GRID_SIZE,
    )
    return clahe.apply(image)


def _fallback_intensity_center(enhanced_diff: np.ndarray):
    """
    Estimate the ROI via intensity centroid when contour-based fitting fails.
    Returns mask, center, target_size; None triplet if intensity is too low.
    """
    weights = enhanced_diff.astype(np.float32)
    background_level = np.percentile(weights, config.WEAK_SIGNAL_BACKGROUND_PERCENTILE)
    weights = np.clip(weights - background_level, 0, None)
    total_intensity = float(weights.sum())

    if total_intensity < config.WEAK_SIGNAL_MIN_TOTAL_INTENSITY:
        print("[segment_roi] Fallback aborted: intensity sum below threshold.")
        return None, None, None

    ys, xs = np.indices(weights.shape)
    center_x = int(np.clip((weights * xs).sum() / total_intensity, 0, weights.shape[1] - 1))
    center_y = int(np.clip((weights * ys).sum() / total_intensity, 0, weights.shape[0] - 1))

    var_x = float((weights * (xs - center_x) ** 2).sum() / total_intensity)
    var_y = float((weights * (ys - center_y) ** 2).sum() / total_intensity)
    axis_x = int(
        np.clip(
            np.sqrt(max(var_x, 0.0)) * config.WEAK_SIGNAL_STD_SCALE,
            config.WEAK_SIGNAL_MIN_AXIS,
            weights.shape[1] // 2,
        )
    )
    axis_y = int(
        np.clip(
            np.sqrt(max(var_y, 0.0)) * config.WEAK_SIGNAL_STD_SCALE,
            config.WEAK_SIGNAL_MIN_AXIS,
            weights.shape[0] // 2,
        )
    )

    fallback_mask = np.zeros_like(enhanced_diff, dtype=np.uint8)
    cv2.ellipse(
        fallback_mask,
        (center_x, center_y),
        (axis_x, axis_y),
        0,
        0,
        360,
        255,
        -1,
    )

    return fallback_mask, (center_x, center_y), config.UNIFIED_CROP_SIZE


def _compute_binary_mask(diff_image: np.ndarray):
    """Generate a binary mask via percentile-based thresholding."""
    positive_pixels = diff_image[diff_image > 0]
    if positive_pixels.size == 0:
        return None

    percentile_val = np.percentile(
        positive_pixels,
        config.DIFF_THRESHOLD_PERCENTILE,
    )
    threshold_val = max(percentile_val, config.DIFF_THRESHOLD_MIN_VALUE)

    _, binary = cv2.threshold(diff_image, threshold_val, 255, cv2.THRESH_BINARY)
    if cv2.countNonZero(binary) == 0:
        _, binary = cv2.threshold(diff_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if cv2.countNonZero(binary) == 0:
            return None

    return binary.astype(np.uint8)


def _refine_center_with_mask(enhanced_diff: np.ndarray, mask: np.ndarray):
    """Refine ROI center inside mask using intensity centroid if enabled."""
    if not config.REFINE_CENTER_ENABLED:
        return None
    weights = enhanced_diff.astype(np.float32)
    scale_mask = mask.astype(np.float32) / 255.0
    weights *= scale_mask
    total_intensity = float(weights.sum())
    if total_intensity < config.REFINE_CENTER_MIN_INTENSITY:
        return None
    if total_intensity > config.REFINE_CENTER_MAX_INTENSITY:
        return None

    ys, xs = np.indices(weights.shape)
    center_x = int(np.clip((weights * xs).sum() / total_intensity, 0, weights.shape[1] - 1))
    center_y = int(np.clip((weights * ys).sum() / total_intensity, 0, weights.shape[0] - 1))
    return (center_x, center_y), total_intensity


def segment_roi(img_background, img_final, blur_kernel, open_kernel_size, close_kernel_size):
    """
    Compute the ROI by comparing background and final frames.
    Falls back to an intensity centroid when contours are unreliable.
    """
    positive_diff = cv2.subtract(img_final, img_background)
    abs_diff = cv2.absdiff(img_background, img_final)
    diff_image = cv2.max(positive_diff, abs_diff)

    blurred_diff = cv2.GaussianBlur(diff_image, blur_kernel, 0)
    enhanced_diff = _apply_contrast_enhancement(blurred_diff)

    binary_mask = _compute_binary_mask(blurred_diff)
    if binary_mask is None:
        print("[segment_roi] Percentile threshold produced empty mask; fallback to intensity centroid.")
        return _fallback_intensity_center(enhanced_diff)

    open_kernel = np.ones(open_kernel_size, np.uint8)
    close_kernel = np.ones(close_kernel_size, np.uint8)
    opened_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, open_kernel)
    closed_mask = cv2.morphologyEx(opened_mask, cv2.MORPH_CLOSE, close_kernel)

    mask_area = cv2.countNonZero(closed_mask)
    image_area = closed_mask.shape[0] * closed_mask.shape[1]
    if mask_area == 0:
        print("[segment_roi] Binary mask emptied after morphology; fallback to intensity centroid.")
        return _fallback_intensity_center(enhanced_diff)

    area_ratio_pixels = mask_area / float(image_area)
    if area_ratio_pixels > config.SEGMENTATION_MAX_AREA_RATIO:
        print(
            f"[segment_roi] Segmented mask too large ({area_ratio_pixels:.3f} of image); "
            "fallback to intensity centroid."
        )
        return _fallback_intensity_center(enhanced_diff)

    contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("[segment_roi] No contour found; fallback to intensity centroid.")
        return _fallback_intensity_center(enhanced_diff)

    largest_contour = max(contours, key=cv2.contourArea)

    if len(largest_contour) < 5:
        print("[segment_roi] Contour too small for ellipse fit; fallback to intensity centroid.")
        return _fallback_intensity_center(enhanced_diff)

    contour_area = cv2.contourArea(largest_contour)
    fitted_ellipse = cv2.fitEllipse(largest_contour)
    ellipse_axes = fitted_ellipse[1]
    ellipse_area = np.pi * (ellipse_axes[0] / 2.0) * (ellipse_axes[1] / 2.0)

    if ellipse_area <= 0:
        print("[segment_roi] Ellipse fit has zero area; fallback to intensity centroid.")
        return _fallback_intensity_center(enhanced_diff)

    area_ratio = contour_area / ellipse_area
    print(f"[segment_roi] Ellipse fit confidence: {area_ratio:.2f}")
    if not (config.ELLIPSE_FIT_MIN_RATIO < area_ratio < config.ELLIPSE_FIT_MAX_RATIO):
        print(
            f"[segment_roi] Ellipse fit outside range "
            f"({area_ratio:.2f} vs {config.ELLIPSE_FIT_MIN_RATIO}-{config.ELLIPSE_FIT_MAX_RATIO}); "
            "fallback to intensity centroid."
        )
        return _fallback_intensity_center(enhanced_diff)

    center_coords = (int(fitted_ellipse[0][0]), int(fitted_ellipse[0][1]))
    major_axis = int(fitted_ellipse[1][0] / 2)
    minor_axis = int(fitted_ellipse[1][1] / 2)
    min_axis = config.WEAK_SIGNAL_MIN_AXIS
    max_axis_x = config.UNIFIED_CROP_SIZE[0] // 2
    max_axis_y = config.UNIFIED_CROP_SIZE[1] // 2
    axes_lengths = (
        int(np.clip(major_axis, min_axis, max_axis_x)),
        int(np.clip(minor_axis, min_axis, max_axis_y)),
    )

    final_mask = np.zeros_like(img_final, dtype=np.uint8)
    cv2.ellipse(final_mask, center_coords, axes_lengths, 0, 0, 360, 255, -1)

    refine_result = _refine_center_with_mask(enhanced_diff, final_mask)
    if refine_result:
        refined_center, total_intensity = refine_result
        shift = np.hypot(refined_center[0] - center_coords[0], refined_center[1] - center_coords[1])
        print(
            f"[segment_roi] Refined center shift: {shift:.1f}px (intensity={total_intensity:.0f})"
        )
        if shift > config.REFINE_CENTER_MAX_SHIFT_PIXELS:
            print(
                "[segment_roi] Refined center shifted "
                f"{shift:.1f}px which exceeds {config.REFINE_CENTER_MAX_SHIFT_PIXELS}px; "
                "using fallback centroid instead."
            )
            return _fallback_intensity_center(enhanced_diff)
        center_coords = refined_center
        final_mask = np.zeros_like(img_final, dtype=np.uint8)
        cv2.ellipse(final_mask, center_coords, axes_lengths, 0, 0, 360, 255, -1)

    target_size = config.UNIFIED_CROP_SIZE

    return final_mask, center_coords, target_size


def crop_and_center_roi(images, mask, center_coords, target_size):
    """
    Crop around the detected ROI center and paste into a fixed-size canvas.
    No scaling is applied; content is centered with zero padding.
    """
    height, width = images[0].shape
    center_x, center_y = center_coords
    target_w, target_h = int(target_size[0]), int(target_size[1])

    # Compute crop bounds around the ROI center.
    x1 = max(0, center_x - target_w // 2)
    y1 = max(0, center_y - target_h // 2)
    x2 = min(width, center_x + target_w // 2)
    y2 = min(height, center_y + target_h // 2)

    actual_w = x2 - x1
    actual_h = y2 - y1

    centered_cropped_images = []
    for img in images:
        canvas = np.zeros((target_h, target_w), dtype=img.dtype)
        paste_x = (target_w - actual_w) // 2
        paste_y = (target_h - actual_h) // 2
        canvas[paste_y: paste_y + actual_h, paste_x: paste_x + actual_w] = img[y1:y2, x1:x2]
        centered_cropped_images.append(canvas)

    canvas_mask = np.zeros((target_h, target_w), dtype=mask.dtype)
    paste_x_mask = (target_w - actual_w) // 2
    paste_y_mask = (target_h - actual_h) // 2
    canvas_mask[paste_y_mask: paste_y_mask + actual_h, paste_x_mask: paste_x_mask + actual_w] = mask[y1:y2, x1:x2]

    return centered_cropped_images, canvas_mask
