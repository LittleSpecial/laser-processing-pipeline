# batch_process.py

import os
import re
import argparse
import collections
import cv2
import subprocess
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import argparse
import subprocess
from src.main import main as process_single_point

def group_files_by_point(data_root):
    pattern = re.compile(r'\[P\w\]\s+(?P<point_id>\d{4})\s+D\s+(?P<delay>-?\d+\.?\d*).*?\s+(?P<phase>[ABC]).*\.tiff')
    file_groups = collections.defaultdict(dict)
    # print(f"正在扫描目录: {data_root}")
    for root, _, files in os.walk(data_root):
        for filename in files:
            match = pattern.search(filename)
            if match:
                data = match.groupdict()
                point_id = data['point_id']
                phase = data['phase']
                delay = data['delay']
                file_groups[point_id][phase] = (Path(root) / filename, delay)
    print(f"扫描完成。共找到 {len(file_groups)} 个加工点的数据。")
    return file_groups

def prepare_input_data(group, temp_dir):
    if temp_dir.exists(): shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True)
    try:
        mapping = {'A': 'P1_background.png', 'B': 'P2_signal.png', 'C': 'P3_final.png'}
        for phase, standard_name in mapping.items():
            source_path, _ = group[phase]
            target_path = temp_dir / standard_name
            img_bytes = np.fromfile(str(source_path), dtype=np.uint8)
            img = cv2.imdecode(img_bytes, cv2.IMREAD_GRAYSCALE)
            if img is None: raise IOError(f"无法读取或解码文件: {source_path}")
            cv2.imwrite(str(target_path), img)
        return True
    except (KeyError, IOError) as e:
        print(f"错误: 准备输入数据失败: {e}。跳过此点。")
        return False

def get_delay_and_prefix(group):
    """从文件组中提取延时和用于文件名的前缀。"""
    # 假设A, B, C三相的延时是相同的
    _, delay_str = group['A'] 
    delay = float(delay_str)
    # 将点号替换为p，例如 -2.4 -> neg2p4
    delay_for_fn = str(delay).replace('.', 'p').replace('-', 'neg')
    
    # 从文件名中提取点ID，例如 '0028'
    pattern = re.compile(r'\[P\w\]\s+(?P<point_id>\d{4})')
    match = pattern.search(group['A'][0].name)
    point_id = match.group('point_id') if match else 'unknown'
    
    filename_prefix = f"Point_{point_id}_Delay_{delay_for_fn}"
    return delay, filename_prefix

def main(data_root, output_root):
    results_dir = output_root
    results_dir.mkdir(parents=True, exist_ok=True)
    file_groups = group_files_by_point(data_root)
    if not file_groups: return

    project_root = Path(__file__).resolve().parent
    temp_input_dir = project_root / "temp_pipeline_input"
    
    spectral_results = []
    processed_count = 0

    failed_points = []
    # --- 阶段 1: 自动处理 ---
    # print("\n--- 开始阶段 1: 自动处理所有加工点 ---")
    total_points = len(file_groups)
    for i, (point_id, group) in enumerate(sorted(file_groups.items())):
        print(f"\n--- [ {i + 1} / {total_points} ] 正在自动处理加工点: {point_id} ---")
        
        # FIXED: 填充自动处理的核心逻辑
        if not prepare_input_data(group, temp_input_dir):
            failed_points.append({'id': point_id, 'group': group})
            continue

        try:
            delay, filename_prefix = get_delay_and_prefix(group)
            temp_point_output_dir = results_dir / f"temp_point_{point_id}"
            
            metrics = process_single_point(
                input_dir=temp_input_dir,
                output_dir=temp_point_output_dir,
                filename_prefix=filename_prefix,
                delay_time=delay,
                heatmap_dir=results_dir
            )
            
            if metrics:
                spectral_results.append(metrics)
                processed_count += 1
            else:
                # 如果函数返回 None，说明处理失败
                raise ValueError("处理函数返回None，表示自动分割失败。")

        except Exception as e:
            print(f"错误: 加工点 {point_id} 的主流程处理失败: {e}")
            failed_points.append({'id': point_id, 'group': group})
        #     cmd = [
        #         'python', '-m', 'src.main',
        #         '--input_dir', str(temp_input_dir),
        #         '--output_dir', str(temp_point_output_dir),
        #         '--filename_prefix', filename_prefix,
        #         '--delay', str(delay),
        #         '--heatmap_dir', str(results_dir)
        #     ]
            
        #     print(f"执行命令: {' '.join(cmd)}")
        #     result = subprocess.run(cmd, check=True, capture_output=True, text=True, encoding='utf-8')
            
        #     # 从 main.py 的输出中解析结果
        #     # 假设 main.py 在成功时会打印一行 "METRICS:{...}"
        #     for line in result.stdout.splitlines():
        #         if line.startswith("METRICS:"):
        #             metrics_str = line.replace("METRICS:", "")
        #             metrics = eval(metrics_str) # 使用eval解析字典字符串
        #             spectral_results.append(metrics)
        #             break
            
        #     processed_count += 1
        # except Exception as e:
        #     print(f"错误: 加工点 {point_id} 的主流程处理失败。")
        #     failed_points.append({'id': point_id, 'group': group})
        #     if isinstance(e, subprocess.CalledProcessError):
        #         print("--- 错误信息 ---\n" + e.stderr + "\n----------------")
        #     else:
        #         print(f"--- 错误信息 ---\n{e}\n----------------")

    # --- 阶段 2: 手动处理失败的点 ---
    if failed_points:
        print("\n\n--- 开始阶段 2: 手动处理失败的加工点 ---")
        # ... (这部分逻辑是正确的，但需要添加结果收集) ...
        for point_info in failed_points:
            point_id = point_info['id']
            group = point_info['group']
            
            while True:
                try:
                    user_input = input(f"\n请输入加工点 '{point_id}' 的中心坐标 (格式: X,Y) 或按Enter跳过: ")
                    if not user_input:
                        print(f"已跳过加工点 {point_id}。")
                        break

                    x_str, y_str = user_input.split(',')
                    center_x = int(x_str.strip())
                    center_y = int(y_str.strip())

                    print(f"收到坐标: X={center_x}, Y={center_y}。正在重新处理...")
                    
                    if not prepare_input_data(group, temp_input_dir):
                        print("准备数据失败，无法继续手动处理。")
                        break

                    delay, filename_prefix = get_delay_and_prefix(group)
                    temp_point_output_dir = results_dir / f"temp_point_{point_id}"
                    
                    metrics = process_single_point(
                        input_dir=temp_input_dir,
                        output_dir=temp_point_output_dir,
                        filename_prefix=filename_prefix,
                        delay_time=delay,
                        manual_center_x=center_x,
                        manual_center_y=center_y,
                        heatmap_dir=results_dir
                    )

                    if metrics:
                        print(f"加工点 {point_id} 手动处理成功。")
                        spectral_results.append(metrics)
                        processed_count += 1
                        break # 成功，跳出while循环
                    else:
                        print(f"错误: 加工点 {point_id} 手动处理失败，请检查坐标。")
                    # cmd = [
                    #     'python', '-m', 'src.main',
                    #     '--input_dir', str(temp_input_dir),
                    #     '--output_dir', str(temp_point_output_dir),
                    #     '--filename_prefix', filename_prefix,
                    #     '--delay', str(delay),
                    #     '--manual_center_x', str(center_x),
                    #     '--manual_center_y', str(center_y),
                    #     '--heatmap_dir', str(results_dir)
                    # ]
                    
                    # result = subprocess.run(cmd, check=True, capture_output=True, text=True, encoding='utf-8')
                    
                    # print(f"加工点 {point_id} 手动处理成功。")
                    
                    # # FIXED: 添加手动处理成功后的结果收集
                    # for line in result.stdout.splitlines():
                    #     if line.startswith("METRICS:"):
                    #         metrics_str = line.replace("METRICS:", "")
                    #         metrics = eval(metrics_str)
                    #         spectral_results.append(metrics)
                    #         break
                    
                    # processed_count += 1
                    # break 

                except ValueError:
                    print("输入格式错误，请输入两个由逗号分隔的数字 (例如: 512, 512)。请重试。")
                except Exception as e:
                    print(f"错误: 加工点 {point_id} 手动处理失败。请检查坐标或错误日志。")
                    if isinstance(e, subprocess.CalledProcessError):
                        print("--- 错误信息 ---\n" + e.stderr + "\n----------------")
                    else:
                        print(f"--- 错误信息 ---\n{e}\n----------------")


    # --- CHANGED: 所有点处理完毕后，保存为Excel文件 ---
    if spectral_results:
        # print("\n--- 所有加工点处理完毕，开始生成Excel数据文件 ---")
        df = pd.DataFrame(spectral_results)
        # 优化: 按照延时的绝对值从小到大排序
        df = df.sort_values(by='abs_delay_for_sort')
        # 删除用于排序的辅助列
        df = df.drop(columns=['abs_delay_for_sort'])
        
        excel_filepath = results_dir / "spectral_data.xlsx"
        try:
            df.to_excel(excel_filepath, index=False, engine='openpyxl')
            print(f"Excel文件已成功保存到: {excel_filepath}")
        except Exception as e:
            print(f"错误: 保存Excel文件失败: {e}")
    # ---------------------------------------------
    
    for item in results_dir.iterdir():
        if item.is_dir() and item.name.startswith("temp_point_"):
            shutil.rmtree(item)
    print(f"\n批处理完成！共成功处理 {processed_count} / {len(file_groups)} 个加工点。")
    print(f"所有结果保存在: {output_root}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="激光加工图像数据批处理脚本")
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--output_root', type=str, required=True)
    args = parser.parse_args()
    main(Path(args.data_root), Path(args.output_root))