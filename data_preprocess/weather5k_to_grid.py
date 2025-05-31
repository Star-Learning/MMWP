import os
import glob
import numpy as np
import xarray as xr
from scipy.spatial import cKDTree
import multiprocessing as mp
from tqdm import tqdm
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


# 1. 自定义网格生成函数
def generate_custom_grid(lat_size=128, lon_size=256):
    """生成自定义尺寸的经纬度网格"""
    lat = np.linspace(90, -90, lat_size)
    lon = np.linspace(0, 360, lon_size, endpoint=False)
    return lon, lat

# 2. 反距离加权插值函数
def idw_interpolation(station_points, station_values, grid_lon, grid_lat, power=2, k=8):
    """反距离加权插值"""
    grid_lon_mesh, grid_lat_mesh = np.meshgrid(grid_lon, grid_lat)
    grid_points = np.column_stack([grid_lon_mesh.ravel(), grid_lat_mesh.ravel()])
    
    tree = cKDTree(station_points)
    distances, indices = tree.query(grid_points, k=k)
    
    distances = np.where(distances == 0, 1e-10, distances)
    weights = 1.0 / (distances ** power)
    weights /= weights.sum(axis=1)[:, np.newaxis]
    
    interpolated_values = np.sum(weights * station_values[indices], axis=1)
    return interpolated_values.reshape(len(grid_lat), len(grid_lon))

# 3. 处理单个NPY文件的函数
def process_npy_file(npy_file, variable_index=0):
    try:
        data = np.load(npy_file, allow_pickle=True).item()
        lat = data['lat'] * 180 - 90
        lon = data['lon'] * 360 - 180
        times = data['date']
        values = data['data'][:, variable_index]
        return pd.DataFrame({
            'DATE': times,
            'TMP': values,
            'lat': lat,
            'lon': lon
        })
    except Exception as e:
        print(f"Error processing {npy_file}: {str(e)}")
        return None

# 4. 主处理函数（支持自定义时间步长）
def process_npy_data(npy_folder, output_dir, lat_size=128, lon_size=256,
                    start_year=2014, end_year=2014, variable_index=0,
                    time_step='6H'):
    """
    处理所有NPY文件并进行插值到自定义网格和时间步长
    
    参数:
    - time_step: 时间步长，支持Pandas频率字符串如'6H','3H','D'等
    """
    # 获取所有NPY文件
    npy_files = glob.glob(os.path.join(npy_folder, '*.npy'))
    print(f"Found {len(npy_files)} NPY files to process")
    
    # 使用多进程处理
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = list(tqdm(
            pool.imap(process_npy_file, npy_files), 
            total=len(npy_files),
            desc="Processing NPY files"
        ))
    
    # 合并所有有效结果
    all_data = pd.concat([r for r in results if r is not None])
    
    # 生成自定义网格
    target_lon, target_lat = generate_custom_grid(lat_size, lon_size)
    print(f"\nUsing custom grid: {lat_size}x{lon_size} (lat x lon)")
    
    # 按年份分组处理
    for year in range(start_year, end_year + 1):
        print(f"\nProcessing year {year} with time step {time_step}...")
        year_data = all_data[all_data['DATE'].dt.year == year].copy()
        
        if len(year_data) == 0:
            print(f"No data found for year {year}")
            continue
            
        # 生成规则时间序列
        start_date = f"{year}-01-01"
        end_date = f"{year}-12-31 23:59:59"
        regular_times = pd.date_range(start=start_date, end=end_date, freq=time_step)
        
        # 准备存储插值结果的数组
        n_time = len(regular_times)
        interpolated_values = np.full((n_time, lat_size, lon_size), np.nan)
        
        # 对每个时间点进行插值
        for i, target_time in enumerate(tqdm(regular_times, desc=f"Interpolating {year}")):
            # 找到时间窗口内的数据 (前后3小时)
            time_window = year_data[
                (year_data['DATE'] >= target_time - pd.Timedelta('6H')) &
                (year_data['DATE'] <= target_time + pd.Timedelta('6H'))
            ]
            
            if len(time_window) < 3:  # 至少需要3个站点才能插值
                continue
                
            # 准备站点数据
            points = time_window[['lon', 'lat']].values
            values = time_window['TMP'].values
            
            # 执行IDW插值
            interpolated_values[i] = idw_interpolation(
                points, values, target_lon, target_lat
            )
        
            # 替代 NetCDF 保存为 .npy 格式
            output_file = os.path.join(
                output_dir, 
                f'station_interpolated_{lat_size}x{lon_size}_{time_step}_temperature_{year}.npy'
            )

            # 构造保存的内容
            output_dict = {
                'temperature': interpolated_values,        # shape: (time, lat, lon)
                'time': regular_times.to_numpy(),          # datetime64 array
                'latitude': target_lat,
                'longitude': target_lon,
                'meta': {
                    'description': f'Station data interpolated to {lat_size}x{lon_size} grid',
                    'interpolation_method': f'IDW (power=2, k=8) with {time_step} time step',
                    'time_step': time_step,
                    'variable': 'TMP' if variable_index == 0 else f'Variable index {variable_index}'
                }
            }

            # 保存
            np.save(output_file, output_dict)
            print(f"Saved interpolated data for {year} to {output_file}")


npy_folder = '/data/zyd/data/weather_5k/WEATHER-5K/global_weather_stations_npy_2'
output_dir = '/data/zyd/data/weather_5k/WEATHER-5K/global_weather_stations_grid_1h'


# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 处理数据 (128x256网格)
process_npy_data(
    npy_folder=npy_folder,
    output_dir=output_dir,
    lat_size=128,      # 纬度方向128个点
    lon_size=256,      # 经度方向256个点
    start_year=2014,
    end_year=2018,
    variable_index=0,   # 0对应TMP温度
    time_step='1H'  # 可设置为'3H','1H','D'等
)