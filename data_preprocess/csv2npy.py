import pandas as pd
import numpy as np
import os
from tqdm import tqdm

def process_csv_to_npy(root_dir, save_dir):
    """
    处理指定目录中的所有CSV文件，将所有数据处理并保存为npy文件。

    参数：
    root_dir (str): CSV文件所在的根目录。
    save_dir (str): 保存npy文件的目录。
    """
    # 确保保存目录存在
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 获取所有CSV文件
    all_files = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith('.csv')]

    for file_path in tqdm(all_files):
        
        # 读取CSV文件，读取所有列
        df = pd.read_csv(file_path, parse_dates=['DATE'])
        # 提取站点名称（去掉文件扩展名）
        basename = os.path.basename(file_path).replace(".csv", "")
        # 保存整个DataFrame为npy格式（包括所有列）
        np.save(os.path.join(save_dir, f"{basename}.npy"), df.values)

# 使用示例
root_dir = '/home/jovyan/work/data/WEATHER-5K/global_weather_stations'  # CSV文件所在目录
save_dir = '/home/jovyan/work/data/WEATHER-5K/global_weather_stations_npy'  # 保存npy文件的目录

# 调用函数
process_csv_to_npy(root_dir, save_dir)