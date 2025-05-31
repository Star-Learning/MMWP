import os
import torch
from torch.utils.data import Dataset, DataLoader
import xarray as xr
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm
import warnings
import glob
import polars as pl
from concurrent.futures import ThreadPoolExecutor

class Weather5kDataset(Dataset):
    def __init__(self,
                 root_dir,
                 input_variables,
                 target_variables,
                 time_window,
                 pred_window, 
                 time_slice=None,
                 step=6,
                 adj_path=None):
        """
        初始化 Weather5k 数据集。

        参数:
        - root_dir (str): 数据根目录，包含所有站点的 CSV 文件。
        - input_variables (list): 输入变量的名称列表。
        - target_variables (list): 目标变量的名称列表，默认为输入变量。
        - time_window (int): 输入时间窗口的大小（时间步数）。
        - pred_window (int): 预测时间窗口的大小（时间步数）。
        - time_slice (tuple): 时间范围 (start, end)。
        - step (int): 时间步长。
        """
        self.root_dir = root_dir
        self.input_variables = input_variables
        self.target_variables = target_variables
        self.time_window = time_window
        self.pred_window = pred_window
        self.time_slice = time_slice
        self.step = step

        self.adj = None
        if adj_path is not None:
            # 加载稀疏邻接矩阵
            sparse_adj = scipy.sparse.load_npz(adjacency_matrix_path)

            # 转换为 n*n 的密集矩阵
            dense_adj = sparse_adj.toarray()
            self.adj = np.load(adj_path, allow_pickle=True)
            print('adj shape:', self.adj.shape)

        # 加载站点数据
        self.all_data, self.lat_lon_coords, self.time_stamps= self._load_variable_data(self.input_variables)
        
    def dates_to_tensor(self, date_list, fmt="%Y-%m-%d-%H-%M-%S"):
        # 1. 转换为 datetime 对象
        datetimes = [datetime.strptime(d, fmt) for d in date_list]
        # 2. 转换为 Unix 时间戳（秒）
        timestamps = [dt.timestamp() for dt in datetimes]
        # 3. 转换为 torch tensor
        return torch.tensor(timestamps, dtype=torch.float32)
    

    def _load_variable_data(self, vars):
        """
        读取预处理后的npz文件，根据变量名vars和时间筛选进行加载。
        """
        all_files = glob.glob(os.path.join(self.root_dir, '*.npy'))

        all_data = []
        lat_lon_coords = []
        time_index = None

        for file_path in tqdm(all_files, desc="Loading .npy station data"):
            # 加载npy文件
            npy_data = np.load(file_path, allow_pickle=True).item()  # 使用 `.item()` 获取字典

            data = npy_data['data']  # 变量数据
            dates = pd.to_datetime(npy_data['date'])  # 日期
            lat = npy_data['lat'].item()  # 纬度
            lon = npy_data['lon'].item()  # 经度
            var_names = npy_data['var_names'].tolist()  # 变量名列表

            # 判断需要的变量索引
            try:
                var_indices = [var_names.index(v) for v in vars]
            except ValueError as e:
                print("判断需要的变量索引: ",file_path)
                continue  # 如果没有这些变量则跳过该站点

            # 提取对应变量数据
            data = data[:, var_indices]

            # 时间筛选
            mask = (dates >= self.time_slice[0]) & (dates < self.time_slice[1])
            if not mask.any():
                print("时间筛选: ",file_path)
                continue

            selected_data = data[mask]
            selected_dates = dates[mask]

            if time_index is None:
                time_index = selected_dates
            else:
                if not selected_dates.equals(time_index):
                    print("selected_dates: ",file_path)
                    continue

            all_data.append(selected_data)
            lat_lon_coords.append([lat, lon])

        if not all_data:
            raise ValueError("没有满足时间筛选的数据")

        # 转换为torch张量
        all_data = torch.from_numpy(np.stack(all_data, axis=1))  # (T, N, C)
        lat_lon_coords = torch.tensor(lat_lon_coords, dtype=torch.float32)  # (N, 2)
        time_stamps = self.dates_to_tensor(time_index.strftime('%Y-%m-%d-%H-%M-%S').tolist())

        return all_data, lat_lon_coords, time_stamps

    def __len__(self):
        return self.all_data.shape[0] - self.time_window - self.pred_window + 1

    def __getitem__(self, idx):
        """
        获取指定索引的数据。

        参数:
        - idx (int): 索引。

        返回:
        - tuple: 输入张量、输出张量、时间戳张量和经纬度坐标张量。
        """
        input_tensors = self.all_data[idx:idx+self.time_window]
        output_tensors = self.all_data[idx+self.time_window:idx+self.time_window+self.pred_window]
        input_timestamps = self.time_stamps[idx:idx+self.time_window]
        # output_timestamps = self.time_stamps[idx+self.time_window:idx+self.time_window+self.pred_window]
        lat_lon = self.lat_lon_coords
        adj_tensor = torch.from_numpy(self.adj)

        return input_tensors, output_tensors, adj_tensor, input_timestamps, lat_lon
    
        # result = {
        #     "input_tensors": input_tensors,
        #     "output_tensors": output_tensors,
        #     "input_timestamps": input_timestamps,
        #     "output_timestamps": output_timestamps,
        #     "lat_lon": lat_lon
        # }

        # return result


if __name__ == "__main__":
    # Weather5k 数据集参数
    weather5k_params = {
        "root_dir": '/data/zyd/data/weather_5k/WEATHER-5K/global_weather_stations_npy_2',
        "input_variables": ["TMP"],
        "target_variables": ['TMP'],
        "time_window": 6,
        "pred_window": 6,
        "time_slice": ("2014-01-01", "2014-01-31"),
        "step": 1,
        "adj_path": '/data/zyd/data/weather_5k/WEATHER-5K/normalized_adj.npy'
    }

    # 初始化 Weather5k 数据集
    weather5k_dataset = Weather5kDataset(**weather5k_params)
    train_loader = DataLoader(weather5k_dataset, batch_size=1, num_workers=2, shuffle=True)

    # 测试 Weather5k 数据加载器
    for i, data in enumerate(train_loader):
        print("input_tensors shape:", data[0].shape)  
        print("output_tensors shape:",  data[1].shape) 
        break