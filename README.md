# 激光加工过程图像自动分析流程 (Laser Processing Image Analysis Pipeline)

本项目旨在提供一个全自动的图像处理与数据分析流程，用于量化分析激光与材料相互作用过程的时间序列图像。

## 功能特性

- **全自动批处理**: 可自动扫描指定文件夹，处理数百个符合命名规则的数据点。
- **图像自动配准**: 使用ORB特征匹配算法，精确对齐不同阶段的图像，消除设备振动等误差。
- **智能ROI分割**: 基于背景（P1）与终态（P3）的物理变化差异，结合Otsu阈值法和轴对齐椭圆拟合，精确、稳健地定位加工区域。
- **动态居中裁剪**: 根据识别出的ROI大小自动计算裁剪区域，并可通过参数控制边缘“留白”大小。
- **量化分析**:
  - 计算信号（P2）相对于背景（P1）的**相对变化率 `(P2-P1)/P1`**。
  - 生成可视化的灰度/伪彩热图。
  - 提取ROI的几何、强度等关键指标，并保存为JSON文件。
- **数据汇总导出**: 将所有时间点的“平均相对变化率”数据汇总并按时间排序，最终导出为 **Excel (.xlsx)** 文件，便于后续在Origin、MATLAB等专业软件中进行分析和绘图。
- **高度可配置**: 核心算法的参数均在 `src/config.py` 文件中开放，方便用户根据不同实验数据进行调整。

## 项目结构

```
laser_processing_pipeline/
├── data/
│   ├── raw/
│   │   └── [your_experiment_folder]/  <-- 将您的原始TIFF图像放在这里
│   └── processed/                    <-- 所有处理结果将保存在这里
├── src/
│   ├── config.py                     <-- 项目的“控制面板”
│   ├── image_alignment.py
│   ├── roi_segmentation.py
│   ├── data_analysis.py
│   ├── main.py
│   └── __init__.py
├── batch_process.py                  <-- 主执行脚本
├── requirements.txt                  <-- 项目依赖库
└── README.md                         <-- 本说明文件
```

## 如何使用

### 1. 环境设置

建议使用Python虚拟环境以避免库版本冲突。

```bash
# 创建一个名为 venv 的虚拟环境
python -m venv venv

# 激活虚拟环境
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### 2. 安装依赖

在激活虚拟环境后，使用 `pip` 安装所有必需的库。

```bash
pip install -r requirements.txt
```

### 3. 准备数据

将您包含 `[P0] XXXX ... A+.tiff` 等文件的实验数据文件夹，整个放入 `data/raw/` 目录下。

### 4. 运行流程

本项目通过 `batch_process.py` 脚本启动。您需要提供两个路径：原始数据所在的根目录和您希望保存结果的根目录。

打开终端，确保您位于 `laser_processing_pipeline` 项目的根目录下，然后执行以下命令：

```bash
python batch_process.py --data_root "path/to/your/raw_data" --output_root "path/to/your/processed_data"
```

**示例 (根据我们之前的交流):**
```bash
python batch_process.py --data_root "data/部分Cu/13.13/22580002" --output_root "data/processed/部分Cu_13_13_22580002_results"
```

## 输出结果说明

处理完成后，您将在指定的输出根目录 (`output_root`) 中看到：

1.  **`spectral_data.xlsx`**: 一个Excel文件，包含了所有加工点的 `point_id`, `delay` (延时), 和 `mean_relative_change` (平均相对变化率)，并已按时间排序。
2.  **`point_XXXX_results/`**: 每个加工点对应的独立结果文件夹。

在每个 `point_XXXX_results` 文件夹内，您会找到：
- **`..._roi_centered.png`**: 裁剪并居中后的P1, P2, P3阶段图像。
- **`..._roi_mask_centered.png`**: 裁剪并居中后的椭圆掩码。
- **`..._relative_change_heatmap.png`**: (P2-P1)/P1 变化率热图。
- **`..._log_ratio_heatmap_centered.png`**: Log(P2/P1) 变化率热图。
- **`..._quantitative_metrics.json`**: 包含该点详细几何和强度指标的JSON文件。

## 参数配置与自定义

本项目最核心的参数都集中在 `src/config.py` 文件中，您可以像修改“设置”一样调整它们。

### ROI分割与裁剪

- **`CROP_PADDING_PIXELS`**: 控制裁剪区域的“边距”。这个值越大，裁剪出的图像中包含的背景区域就越多。

### 热图外观

您可以完全控制 `(P2-P1)/P1` 热图的显示效果。

- **`REL_CHANGE_VMIN` / `REL_CHANGE_VMAX`**:
  控制热图色标的最小值和最大值。例如，设置为 `0.0` 和 `0.5` 意味着任何低于0%的变化都显示为最暗色，任何高于50%的变化都显示为最亮色。

- **`REL_CHANGE_HEATMAP_COLORMAP`**:
  控制热图的配色方案。您可以将其改为 `'gray'` (灰度), `'coolwarm'` (红-白-蓝), `'hot'` (黑-红-黄), `'viridis'` (科研常用色)等。