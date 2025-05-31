import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm

import xarray as xr
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

def compute_means_stds_weatherbench(root_dir, save_fp):
    """
    计算多个 WeatherBench NetCDF 文件中每个变量的全局均值和标准差。

    参数：
        nc_files: list of str，NetCDF 文件路径列表

    返回：
        means: dict，变量名到其均值的映射
        stds: dict，变量名到其标准差的映射
    """
    # 打开并合并多个 NetCDF 文件
    nc_files = glob.glob(os.path.join(root_dir, '*.nc'))
    ds = xr.open_mfdataset(nc_files, combine='by_coords',engine="netcdf4")

    means = {}
    stds = {}

    for var in tqdm(ds.data_vars):
        # 计算每个变量在时间、纬度和经度维度上的均值和标准差
        means[var] = float(ds[var].mean(dim=['time', 'lat', 'lon'], skipna=True).values)
        stds[var] = float(ds[var].std(dim=['time', 'lat', 'lon'], skipna=True).values)

    with open(save_fp, 'w') as f:
        f.write("Variable\tMean\tStd\n")
        for var in means:
            f.write(f"{var}\t{means[var]}\t{stds[var]}\n")
    return means, stds


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


def preprocess_and_save_npy(root_dir, save_dir, means, stds):
    """
    将每个CSV文件中所有变量（除了经纬度和时间）使用给定的均值和标准差进行归一化，并保存为npy。
    """
    os.makedirs(save_dir, exist_ok=True)

    all_files = glob.glob(os.path.join(root_dir, '*.csv'))

    for file_path in tqdm(all_files, desc="Preprocessing and saving .npy"):
        save_fp = os.path.join(save_dir, os.path.basename(file_path).replace('.csv', '.npy'))
        if os.path.exists(save_fp):
            continue

        df = pd.read_csv(file_path, parse_dates=['DATE'])
        if df.empty:
            continue

        # 获取所有变量名（排除固定字段）
        exclude = ['DATE', 'LATITUDE', 'LONGITUDE', 'MASK', 'TIME_DIFF']
        variable_cols = [col for col in df.columns if col not in exclude]

        # 经纬度归一化
        lat = (df['LATITUDE'].iloc[0] + 90) / 180
        lon = (df['LONGITUDE'].iloc[0] + 180) / 360

        # 使用给定的均值和标准差进行归一化
        df[variable_cols] = (df[variable_cols] - means) / (stds + 1e-9)

        # 保存为npy文件
        np.save(
            save_fp,
            {
                'data': df[variable_cols].to_numpy(dtype=np.float32),
                'date': df['DATE'].to_numpy(dtype='datetime64[s]'),
                'lat': np.float32(lat),
                'lon': np.float32(lon),
                'var_names': np.array(variable_cols)
            }
        )


def convert_weatherbench_to_graph_npy(root_dir, save_dir, means, stds):
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. 加载数据集时优化分块策略
    nc_files = glob.glob(os.path.join(root_dir, '*.nc'))
    ds = xr.open_mfdataset(
        nc_files,
        combine='by_coords',
        engine="netcdf4",
        chunks='auto'  # 关键优化：按空间分块，时间维度不分块
    ).transpose('time', 'lat', 'lon')  # 确保维度顺序一致
    
    # 2. 预加载常用数据到内存
    var_names = list(ds.data_vars.keys())
    time_values = ds.time.values.astype('datetime64[s]')
    lats = ds.lat.values
    lons = ds.lon.values
    
    # 3. 预计算归一化参数（向量化）
    means_arr = np.array([means[v] for v in var_names], dtype=np.float32)
    stds_arr = np.array([stds[v] for v in var_names], dtype=np.float32) + 1e-9
    
    # 4. 按纬度带处理优化内存访问
    for lat_idx, lat in enumerate(tqdm(lats, desc="Processing latitudes")):
        # 一次性加载整个纬度带数据到内存（关键优化！）
        lat_slice = ds.isel(lat=lat_idx).load()
        
        # 预计算纬度归一化值
        lat_norm = np.float32((lat + 90) / 180)
        
        for lon_idx, lon in enumerate(lons):
            save_path = os.path.join(save_dir, f"{lat:.2f}_{lon:.2f}.npy")
            if os.path.exists(save_path):
                continue
            
            # 5. 向量化获取所有变量数据
            point_data = lat_slice.isel(lon=lon_idx).to_array(dim='var')
            data = point_data.transpose('time', 'var').values.astype(np.float32)
            
            # 向量化归一化计算
            normalized_data = (data - means_arr) / stds_arr
            
            # 处理经度归一化
            lon_norm = np.float32((lon % 360) / 360)

            if 't2m' in var_names:
                var_names[var_names.index('t2m')] = 'TMP'  # 替换变量名
            # 6. 优化保存格式（使用allow_pickle=False加速）
            np.save(
                save_path,
                {
                    'data': normalized_data,
                    'date': time_values,
                    'lat': lat_norm,
                    'lon': lon_norm,
                    'var_names': np.array(var_names)
                },
            )

    ds.close()
