import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
import xarray as xr

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

def calculate_mean_std_weather5k(root_dir, save_path="mean_std.txt"):
    """
    计算数据集中所有变量的均值和标准差，并保存到txt文件。
    """
    all_files = glob.glob(os.path.join(root_dir, '*.csv'))
    exclude = ['DATE', 'LATITUDE', 'LONGITUDE', 'MASK', 'TIME_DIFF']
    variable_sums = None
    variable_squares = None
    total_count = 0

    for file_path in tqdm(all_files, desc="Calculating mean and std"):
        df = pd.read_csv(file_path)
        if df.empty:
            continue

        variable_cols = [col for col in df.columns if col not in exclude]
        data = df[variable_cols].to_numpy(dtype=np.float32)

        if variable_sums is None:
            variable_sums = np.zeros(data.shape[1], dtype=np.float64)
            variable_squares = np.zeros(data.shape[1], dtype=np.float64)

        variable_sums += np.sum(data, axis=0)
        variable_squares += np.sum(data ** 2, axis=0)
        total_count += data.shape[0]

    means = variable_sums / total_count
    stds = np.sqrt(variable_squares / total_count - means ** 2)

    # 保存均值和标准差到txt文件
    with open(save_path, "w") as f:
        f.write("Means:\n")
        f.write(", ".join(map(str, means)) + "\n")
        f.write("Stds:\n")
        f.write(", ".join(map(str, stds)) + "\n")
 
    print(f"Means and stds saved to {save_path}")
    print("Means:", means)
    print("Stds:", stds)
    return means, stds


def calculate_mean_std_weatherbench(root_dir, save_path="mean_std_weatherbench.txt"):
    """
    读取WeatherBench数据集，计算所有变量的均值和标准差，并保存到txt文件。
    """
    all_files = glob.glob(os.path.join(root_dir, '*.nc'))  # 修改为读取 .nc 文件
    variable_sums = None
    variable_squares = None
    total_count = 0

    for file_path in tqdm(all_files, desc="Calculating mean and std for WeatherBench"):
        # 使用 xarray 读取 NetCDF 文件
        ds = xr.open_dataset(file_path)
        variables = list(ds.data_vars.keys())  # 获取所有变量名

        for var in variables:
            data = ds[var].values  # 提取变量数据为 numpy 数组
            if data.ndim > 2:  # 如果数据有时间维度或其他维度，展平为二维
                data = data.reshape(-1, data.shape[-1])

            if variable_sums is None:
                variable_sums = np.zeros(data.shape[1], dtype=np.float64)
                variable_squares = np.zeros(data.shape[1], dtype=np.float64)

            variable_sums += np.nansum(data, axis=0)  # 忽略 NaN 值
            variable_squares += np.nansum(data ** 2, axis=0)
            total_count += np.sum(~np.isnan(data), axis=0)  # 统计非 NaN 值的数量

    means = variable_sums / total_count
    stds = np.sqrt(variable_squares / total_count - means ** 2)

    # 保存均值和标准差到txt文件
    with open(save_path, "w") as f:
        f.write("Means:\n")
        f.write(", ".join(map(str, means)) + "\n")
        f.write("Stds:\n")
        f.write(", ".join(map(str, stds)) + "\n")

    print(f"Means and stds saved to {save_path}")
    print("Means:", means)
    print("Stds:", stds)
    return means, stds
