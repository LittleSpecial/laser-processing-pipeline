# src/plotting.py

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np # 新增导入 numpy
from . import config # 导入config以获取谱线图参数

def plot_spectral_line(results, output_filepath):
    """
    根据所有加工点的结果，绘制变化率随延时变化的谱线图。
    """
    if not results:
        print("警告: 没有可用于绘图的结果数据。")
        return

    df = pd.DataFrame(results)

    # CHANGED: 处理延时数据 - 取绝对值并根据需要反转X轴
    # 如果延时是负数，我们取其绝对值
    df['display_delay'] = df['delay'].apply(lambda x: abs(x))

    # 根据配置决定排序方式
    if config.SPECTRAL_PLOT_INVERT_X_AXIS:
        # 如果需要反转，则按绝对值降序排序，但实际绘图时matplotlib会从小到大画
        # 所以我们直接确保数据是按绝对值递增排列的
        df = df.sort_values(by='display_delay', ascending=True)
    else:
        # 如果不需要反转，则按原始delay值排序
        df = df.sort_values(by='delay', ascending=True)


    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))

    ax.plot(df['display_delay'], df['metric'], marker='o', linestyle='-', label='Mean Relative Change')
    
    ax.set_title('Mean Relative Change Rate vs. Delay Time', fontsize=16)
    ax.set_xlabel('Delay Time (seconds)', fontsize=12) # 明确单位为秒
    ax.set_ylabel('Mean Relative Change ((P2-P1)/P1)', fontsize=12)
    ax.legend()
    
    # 增加网格线
    ax.grid(True, linestyle='--', alpha=0.7)

    # 确保X轴刻度是整洁的
    # ax.set_xticks(np.unique(df['display_delay'])) # 如果延时值不多，可以显示所有延时点
    
    # 考虑添加刻度标签的旋转，如果延时值很多的话
    plt.xticks(rotation=45, ha='right')

    try:
        fig.savefig(output_filepath, dpi=300, bbox_inches='tight')
        print(f"谱线图已成功保存到: {output_filepath}")
    except Exception as e:
        print(f"错误: 保存谱线图失败: {e}")
    
    plt.close(fig)