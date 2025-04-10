import os
import cv2
import numpy as np
import pywt
import time
import matplotlib.pyplot as plt
from typing import List, Tuple
from matplotlib import font_manager

# ---------------------- 多尺度Retinex增强----------------------
def multiscale_retinex(image: np.ndarray, alpha: float = 80.0) -> np.ndarray:
    """改进的MBR增强"""
    h, w = image.shape[:2]
    img = image.astype(np.float32) + 1e-6
    
    min_dim = min(h, w)
    sigma_list = [int(min_dim*0.05), int(min_dim*0.1), int(min_dim*0.2)]
    
    retinex = np.zeros_like(img)
    for sigma in sigma_list:
        if sigma < 1:
            sigma = 1
        blurred = cv2.GaussianBlur(img, (0, 0), sigma)
        retinex += (np.log10(img) - np.log10(blurred)) * (1.0 / len(sigma_list))
    
    retinex = cv2.normalize(retinex, None, 0, 255, cv2.NORM_MINMAX)

    result = cv2.addWeighted(img, 0.5, retinex, 0.5, 0)
    return np.clip(result * (alpha / 100.0), 0, 255).astype(np.uint8)

# ----------------------  小波金字塔处理----------------------
def wavelet_pyramid(image: np.ndarray, level: int = 3, enhance_factor: float = 1.5) -> np.ndarray:
    """
    多级小波分解与增强（真正的小波金字塔实现）
    参数：
        level - 分解层级
        enhance_factor - 高频增强系数
    """

    h, w = image.shape
    if h % 2 != 0 or w % 2 != 0:
        image = cv2.resize(image, (w//2*2, h//2*2))
    
    coeffs = pywt.wavedec2(image.astype(np.float32), 'haar', level=level)

    coeffs_new = [coeffs[0]]
    for i in range(1, len(coeffs)):
        enhanced = [c * enhance_factor for c in coeffs[i]]
        coeffs_new.append(tuple(enhanced))

    reconstructed = pywt.waverec2(coeffs_new, 'haar')

    return cv2.normalize(reconstructed, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# ---------------------- 预处理流程----------------------
def advanced_preprocess(image: np.ndarray) -> np.ndarray:
    """先小波金字塔再Retinex的联合预处理"""

    wavelet_img = wavelet_pyramid(image)

    msr_img = multiscale_retinex(wavelet_img)

    return cv2.convertScaleAbs(msr_img, alpha=1.2, beta=0)

# ---------------------- 特征提取 ----------------------
def extract_features(image: np.ndarray) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
    """修正参数错误的特征提取"""
    h, w = image.shape
    n_features = min(500000, int(h*w))

    detector = cv2.ORB_create(
        nfeatures=n_features,
        fastThreshold=20,
        edgeThreshold=31,
        scaleFactor=1.2,
        patchSize=31,
        scoreType=cv2.ORB_HARRIS_SCORE
    )
    kps = detector.detect(image, None)
    
    descriptor = cv2.BRISK_create(thresh=25, octaves=4)
    
    if kps and descriptor:
        kps, descs = descriptor.compute(image, kps)
    else:
        return [], np.array([])
    
    if len(kps) > 0:
        responses = np.array([kp.response for kp in kps])
        valid_mask = responses > np.percentile(responses, 20)
        return [kp for kp, v in zip(kps, valid_mask) if v], descs[valid_mask]
    return [], np.array([])

# ---------------------- 核心流程---------------------
def process_image_pair(img1_path: str, img2_path: str):
    """最终完整流程"""
    total_start = time.time()
    time_stats = {'preprocess': 0.0, 'feature': 0.0, 'matching': 0.0, 'gms': 0.0, 'ransac': 0.0}
    
    try:
        print(f"\n处理图像对: {os.path.basename(img1_path)} vs {os.path.basename(img2_path)}")

        orig1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
        orig2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
        if orig1 is None or orig2 is None:
            raise ValueError(f"图像读取失败，请检查路径：{img1_path} 或 {img2_path}")

        start = time.time()
        enhanced1 = advanced_preprocess(orig1)
        enhanced2 = advanced_preprocess(orig2)
        time_stats['preprocess'] = time.time() - start

        start = time.time()
        kps1, desc1 = extract_features(enhanced1)
        kps2, desc2 = extract_features(enhanced2)
        time_stats['feature'] = time.time() - start
        print(f"特征点数: {len(kps1)} vs {len(kps2)}")

        compare_features(orig1, enhanced1, kps1, kps1)
        compare_features(orig2, enhanced2, kps2, kps2)

        start = time.time()
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = []
        if len(desc1) > 0 and len(desc2) > 0:
            matches = bf.knnMatch(desc1, desc2, k=2)
            good_matches = []
            for m, n in matches:
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)
            print(f"比率测试后: {len(good_matches)}")
        else:
            print("警告：无有效描述子")
        time_stats['matching'] = time.time() - start

        start = time.time()
        gms_matches = vectorized_gms_filter(kps1, kps2, good_matches, orig1.shape)
        time_stats['gms'] = time.time() - start
        print(f"GMS筛选后: {len(gms_matches)}")

        start = time.time()
        ransac_matches = robust_ransac(kps1, kps2, gms_matches)
        time_stats['ransac'] = time.time() - start
        print(f"RANSAC优化后: {len(ransac_matches)}")

        accuracy = len(ransac_matches) / len(gms_matches) if len(gms_matches) > 0 else 0
        visualize_matches(enhanced1, enhanced2, kps1, kps2, ransac_matches, accuracy)

        total_time = time.time() - total_start
        print(f"\n总处理时间: {total_time:.2f}秒")
        print(f"  预处理: {time_stats['preprocess']:.2f}秒")
        print(f"  特征提取: {time_stats['feature']:.2f}秒")
        print(f"  特征匹配: {time_stats['matching']:.2f}秒")
        print(f"  GMS筛选: {time_stats['gms']:.2f}秒")
        print(f"  RANSAC优化: {time_stats['ransac']:.2f}秒")
        
    except Exception as e:
        print(f"处理出错: {str(e)}")

# ---------------------- 辅助函数 ----------------------
def vectorized_gms_filter(kps1, kps2, matches, img_shape):
    """GMS筛选"""
    h, w = img_shape
    grid_size = max(20, int(min(h, w)*0.03))
    
    pts = np.array([kps1[m.queryIdx].pt for m in matches])
    valid_mask = (pts[:, 0] >= 0) & (pts[:, 0] < w) & (pts[:, 1] >= 0) & (pts[:, 1] < h)
    pts = pts[valid_mask]
    
    grid_indices = (pts // grid_size).astype(int)
    grid_rows, grid_cols = h//grid_size+1, w//grid_size+1
    grid = np.zeros((grid_rows, grid_cols), dtype=int)
    np.add.at(grid, (grid_indices[:, 1], grid_indices[:, 0]), 1)
    
    threshold = max(2, int(np.median(grid[grid > 0]) * 0.5))
    grid_values = grid[grid_indices[:, 1], grid_indices[:, 0]]
    return [m for m, v in zip(np.array(matches)[valid_mask], grid_values) if v >= threshold]

def robust_ransac(kps1, kps2, matches):
    """RANSAC"""
    if len(matches) < 4:
        return []
    
    src_pts = np.float32([kps1[m.queryIdx].pt for m in matches])
    dst_pts = np.float32([kps2[m.trainIdx].pt for m in matches])
    
    distances = np.linalg.norm(src_pts - dst_pts, axis=1)
    threshold = max(1.5, np.median(distances)*0.3)
    
    _, mask = cv2.findHomography(src_pts, dst_pts, cv2.USAC_MAGSAC, threshold, maxIters=2000)
    return [matches[i] for i in np.where(mask.ravel() == 1)[0]]

# ---------------------- 可视化函数 ----------------------
def visualize_keypoints(image: np.ndarray, 
                       kps: List[cv2.KeyPoint], 
                       title: str,
                       ax: plt.Axes):
    display = cv2.drawKeypoints(
        cv2.cvtColor(image, cv2.COLOR_GRAY2BGR),
        kps,
        None,
        color=(0, 255, 0),
        flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS
    )
    ax.imshow(display)
    ax.set_title(f"{title}\n特征点数: {len(kps)}", 
                fontproperties=font_manager.FontProperties(fname='C:/Windows/Fonts/simhei.ttf'))
    ax.axis('off')

def compare_features(orig_img: np.ndarray, 
                    enhanced_img: np.ndarray,
                    orig_kps: List[cv2.KeyPoint],
                    enhanced_kps: List[cv2.KeyPoint]):
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    visualize_keypoints(enhanced_img, enhanced_kps, "预处理图像特征", axes[0])
    visualize_keypoints(orig_img, orig_kps, "原始图像特征", axes[1])
    plt.tight_layout()
    plt.show()

def visualize_matches(img1: np.ndarray, 
                     img2: np.ndarray, 
                     kps1: List[cv2.KeyPoint], 
                     kps2: List[cv2.KeyPoint], 
                     matches: List[cv2.DMatch],
                     accuracy: float):
    display = cv2.drawMatches(
        cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR), kps1,
        cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR), kps2,
        matches, None,
        matchColor=(0, 255, 0),
        singlePointColor=(255, 0, 0),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    
    plt.figure(figsize=(18, 9))
    plt.imshow(display)
    plt.title(f"匹配正确率: {accuracy:.2%} (总匹配数: {len(matches)})", 
             fontproperties=font_manager.FontProperties(fname='C:/Windows/Fonts/simhei.ttf'))
    plt.axis('off')
    plt.show()

def process_dataset(dataset_dir: str):
    """批量处理（保持原样）"""
    img_exts = ('.jpg', '.png', '.jpeg')
    img_paths = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir) if f.lower().endswith(img_exts)]
    
    for i in range(len(img_paths)-1):
        process_image_pair(img_paths[i], img_paths[i+1])

if __name__ == "__main__":
    dataset_directory = "D:/Images"
    process_dataset(dataset_directory)