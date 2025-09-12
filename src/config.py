# src/config.py

from pathlib import Path

# --- 路径配置 ---
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"

HEATMAP_OUTPUT_DIR = PROCESSED_DATA_DIR / "all_relative_change_heatmaps"
# --- 图像配准 (ORB) 参数 ---
ORB_N_FEATURES = 5000
ORB_MATCH_RATIO = 0.75

# --- ROI分割参数 ---
SEGMENTATION_BLUR_KERNEL = (21, 21)
MORPH_OPEN_KERNEL_SIZE = (3, 3)
MORPH_CLOSE_KERNEL_SIZE = (7, 7)
# NEW: 裁剪时在椭圆外接矩形的基础上增加的边距（像素）
CROP_PADDING_PIXELS = 100 # 您可以调整这个值来控制裁剪区域的大小

ELLIPSE_FIT_MIN_RATIO = 0.7
ELLIPSE_FIT_MAX_RATIO = 1.3

UNIFIED_CROP_SIZE = (150, 110)

# --- 数据分析参数 ---
LOG_EPSILON = 1e-9
LOG_RATIO_HEATMAP_COLORMAP = 'inferno'

# --- (P2-P1)/P1 相对变化率热图的参数 ---
REL_CHANGE_P1_THRESHOLD = 10
REL_CHANGE_HEATMAP_COLORMAP = 'gray' 
REL_CHANGE_VMIN = -1.0
REL_CHANGE_VMAX = 1.0

# --- 谱线图参数 ---
SPECTRAL_PLOT_INVERT_X_AXIS = True