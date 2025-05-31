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
import os

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

VARIABLE_NAME_MAPPING = {
    "10m_u_component_of_wind": "u10",
    "10m_v_component_of_wind": "v10",
    "2m_temperature": "t2m",
    "2m_dewpoint_temperature": "d2m",
    "surface_pressure": "sp",
    "geopotential": "z",
    "geopotential_500": "500",
    "potential_vorticity": "pv",
    "relative_humidity": "r",
    "specific_humidity": "s"
}



class WeatherBenchDataset(Dataset):
    def __init__(self,
                 root_dir,
                 input_variables,
                 target_variables,
                 time_window,
                 pred_window,
                 time_slice=None,
                 step=1,
                 transform=False):
        """
        初始化 WeatherBench 数据集。

        参数:
        - root_dir (str): 数据根目录，包含每个变量的子目录。
        - input_variables (list): 输入变量的名称列表。
        - target_variables (list): 目标变量的名称列表。
        - time_window (int): 输入时间窗口的大小（时间步数）。
        - pred_window (int): 预测时间窗口的大小（时间步数）。
        - time_slice (tuple, optional): 时间范围 (start, end)。
        - step (int, optional): 时间步长，默认为1。
        - transform (bool, optional): 是否应用数据变换，默认为False。
        """
        self.root_dir = root_dir
        self.transform = transform
        self.time_window = time_window
        self.pred_window = pred_window
        self.time_slice = time_slice
        self.step = step

        # 映射变量名称
        self.input_variables = input_variables
        self.target_variables = target_variables
        self.actual_input_variables = [VARIABLE_NAME_MAPPING.get(var, var) for var in input_variables]
        self.actual_target_variables = [VARIABLE_NAME_MAPPING.get(var, var) for var in target_variables]

        self.input_data = []
        self.input_variable_channels = []
        self.input_time_stamps = []

        self.target_data = []
        self.target_variable_channels = []
        self.target_time_stamps = []

        # 加载输入变量数据
        for i in range(len(self.input_variables)):
            folder_name = self.input_variables[i]
            var = self.actual_input_variables[i]
            file_path = os.path.join(root_dir, folder_name, '*.nc')
            var_data, time_stamps = self._load_variable_data(var, file_path)
            self.input_data.append(var_data)
            if not self.input_time_stamps:
                self.input_time_stamps = time_stamps
    
        # 加载目标变量数据
        for i in range(len(self.target_variables)):
            folder_name = self.target_variables[i]
            var = self.actual_target_variables[i]
            file_path = os.path.join(root_dir, folder_name, '*.nc')
            var_data, time_stamps = self._load_variable_data(var, file_path)
            self.target_data.append(var_data)
            if not self.target_time_stamps:
                self.target_time_stamps = time_stamps

        self.input_data = torch.cat(self.input_data, dim=1)
        self.target_data = torch.cat(self.target_data, dim=1)

        print(f"Loaded {self.time_slice} time steps for input variables {input_variables} and target variables {target_variables} with step {step}.")

        # normalized  (time, C, H, W)
        self.mean_in = self.input_data.mean(dim=(2, 3), keepdim=True)
        self.mean_out = self.input_data.mean(dim=(2, 3), keepdim=True)
        self.std_in = self.input_data.mean(dim=(2, 3), keepdim=True)
        self.std_out = self.input_data.mean(dim=(2, 3), keepdim=True)

        self.input_data = (self.input_data - self.mean_in) / self.std_in
        self.target_data = (self.target_data - self.mean_out) / self.std_out
        
    def dates_to_tensor(self, date_list, fmt="%Y-%m-%d-%H-%M-%S"):
        # 1. 转换为 datetime 对象
        datetimes = [datetime.strptime(d, fmt) for d in date_list]
        # 2. 转换为 Unix 时间戳（秒）
        timestamps = [dt.timestamp() for dt in datetimes]
        # 3. 转换为 torch tensor
        return torch.tensor(timestamps, dtype=torch.float32)
    

    def _load_variable_data(self, var, file_path):
        """
        加载指定变量的数据。

        参数:
        - var (str): 用户提供的变量名称。
        - mapped_var (str): 映射后的变量名称。
        - file_path (str): 文件路径模式。
        - data_list (list): 存储数据的列表。
        - channels_list (list): 存储通道数的列表。
        """

        if not os.path.exists(self.root_dir):
            raise FileNotFoundError(f"rootdir not found : {self.root_dir}")
        files = glob.glob(file_path)

        if not files:
            raise FileNotFoundError(f"No .nc files found for variable {var} at {file_path}")

        dataset = xr.open_mfdataset(file_path, combine="by_coords", engine="netcdf4")
        # dataset = xr.open_mfdataset(file_path, combine="by_coords")

        if var not in dataset:
            raise KeyError(f"Variable {var} not found in dataset for {file_path}")
        
        # 读取变量数据
        var_data = dataset[var]
        # 按照time_slice筛选
        if self.time_slice:
            # var_data = var_data.where((var_data.time >= np.datetime64(self.time_slice[0])) & (var_data.time < np.datetime64(self.time_slice[1])))
            var_data = var_data.sel(time=slice(*self.time_slice))
        # 按照步长来取
        var_data = var_data.isel(time=slice(0, None, self.step))

        # np.datetime64转换为 '2024-12-01-08-00-00'年月日时分秒的字符串
        time_stamps = var_data.time.values
        time_stamps = [x.astype('datetime64[s]').astype(object).strftime('%Y-%m-%d-%H-%M-%S') for x in time_stamps]
        if len(var_data.shape) == 3:
            var_data = var_data.expand_dims(dim='channel', axis=1)

        var_data = torch.from_numpy(var_data.to_numpy()).type(torch.float32)
        time_stamps = self.dates_to_tensor(time_stamps)
        return var_data, time_stamps

    def __len__(self):
        return self.input_data.shape[0] - self.time_window - self.pred_window + 1


    def __getitem__(self, idx):
        """
        获取指定索引的数据。

        参数:
        - idx (int): 索引。

        返回:
        - tuple: 输入张量、输出张量和时间戳张量。
        """

        input_tensors = self.input_data[idx: idx + self.time_window]
        output_tensors = self.target_data[idx + self.time_window: idx + self.time_window + self.pred_window]
        input_timestamps = self.input_time_stamps[idx: idx + self.time_window]
        # output_timestamps = self.target_time_stamps[idx + self.time_window: idx + self.time_window + self.pred_window]
        
        return input_tensors, output_tensors, input_timestamps
    
        # result = {
        #     "input_tensors": input_tensors,
        #     "output_tensors": output_tensors,
        #     "input_timestamps": input_timestamps,
        #     "output_timestamps": output_timestamps,
        # }
        
        # return result

if __name__ == "__main__":
    root_dir = '/data/zyd/data/all_1.40625deg/2m_temperature/media/rasp/Elements/weather-benchmark/1.40625deg'
    # variables = ["10m_u_component_of_wind",
    #              "10m_u_component_of_wind",
    #              "2m_temperature",
    #              "geopotential",
    #              "potential_vorticity",
    #              "relative_humidity"]  # 如果为 None 则加载所有变量
    
    variables = ["2m_temperature"]  # 如果为 None 则加载所有变量
    time_slice = ("2014-01-01", "2014-01-10")  # 加载指定时间范围的数据
    step = 4  # 每隔 6 个时间步加载一个数据
    time_window = 6
    pred_window = 6
    dataset = WeatherBenchDataset(root_dir, variables, time_window=time_window, pred_window=pred_window, time_slice=time_slice, step=step, transform=None, target_variables=['2m_temperature'])
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # 测试数据加载器
    for data in dataloader:
        print("input_tensors shape:", data['input_tensors'].shape)  # 应该是 (batch_size, T, total_channels, H, W) = (2, 6, 4, 32, 64)
        print("output_tensors shape:",  data['output_tensors'].shape)  # 应该是 (batch_size, T', total_channels, H, W) = (2, 6, 4, 32, 64)
        print(data['input_timestamps'])
        print(data['output_timestamps'])
        break