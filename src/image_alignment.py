import cv2
import numpy as np

def align_images(img_ref, img_to_align, n_features, match_ratio):
    """
    使用ORB特征检测器对齐两个灰度图像。
    
    参数:
    - img_ref (np.array): 参考图像。
    - img_to_align (np.array): 需要被对齐的图像。
    - n_features (int): ORB检测器要寻找的最大特征点数量。
    - match_ratio (float): 用于筛选优质匹配点的比率。
    
    返回:
    - aligned_img (np.array): 对齐后的图像。
    - h_matrix (np.array): 计算出的单应性变换矩阵。
    """
    # 1. 初始化ORB检测器
    orb = cv2.ORB_create(nfeatures=n_features)

    # 2. 寻找关键点和描述符
    keypoints_ref, descriptors_ref = orb.detectAndCompute(img_ref, None)
    keypoints_align, descriptors_align = orb.detectAndCompute(img_to_align, None)

    # 检查是否找到了足够的描述符
    if descriptors_ref is None or descriptors_align is None:
        print("警告: 无法在其中一张图像中找到足够的描述符。")
        return None, None

    # 3. 匹配特征点
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    raw_matches = matcher.knnMatch(descriptors_align, descriptors_ref, k=2)

    # 4. 应用Lowe's Ratio Test筛选优质匹配
    good_matches = []
    for m, n in raw_matches:
        if m.distance < match_ratio * n.distance:
            good_matches.append(m)
    
    print(f"找到 {len(good_matches)} 个优质匹配点。")

    if len(good_matches) > 10: 
        # 5. 计算单应性变换矩阵
        src_pts = np.float32([keypoints_align[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints_ref[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        h_matrix, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # 6. 应用变换矩阵进行图像 warping
        height, width = img_ref.shape
        aligned_img = cv2.warpPerspective(img_to_align, h_matrix, (width, height))
        
        return aligned_img, h_matrix
    else:
        print("警告: 没有找到足够的优质匹配点来进行图像对齐。")
        return None, None