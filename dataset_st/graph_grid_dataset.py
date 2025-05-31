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
import concurrent.futures
import scipy.sparse as sp
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


class GraphDatasetAll(Dataset):
    def __init__(self,
                 root_dirs,
                 input_variables,
                 target_variables,
                 time_window,
                 pred_window, 
                 time_slice=None,
                 step=6,
                 adj_path=None,
                 target_nodes=5671):
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
        self.root_dirs = root_dirs
        self.input_variables = input_variables
        self.target_variables = target_variables
        self.time_window = time_window
        self.pred_window = pred_window
        self.time_slice = time_slice
        self.step = step
        self.target_nodes = target_nodes

        self.adj = None
        if adj_path is not None:
            A = sp.load_npz(adj_path).tocoo()  # 转为COO格式（方便转换）
            indices = torch.tensor([A.row, A.col], dtype=torch.long)
            values = torch.tensor(A.data, dtype=torch.float32)
            shape = A.shape
            self.adj = torch.sparse.FloatTensor(indices, values, torch.Size(shape))
            # self.adj = np.load(adj_path, allow_pickle=True)
            print('adj shape:', self.adj.shape)

        # if adj_path is not None:
        #     A = sp.load_npz(adj_path).tocoo()  # 转为COO格式（方便转换）
        #     # 只取行列前20个
        #     mask = (A.row < 20) & (A.col < 20)
        #     indices = torch.tensor([A.row[mask], A.col[mask]], dtype=torch.long)
        #     values = torch.tensor(A.data[mask], dtype=torch.float32)
        #     shape = (20, 20)  # 限制为前20行20列
        #     self.adj = torch.sparse.FloatTensor(indices, values, torch.Size(shape))
        #     print('adj shape:', self.adj.shape)

        # 加载站点数据
        self.all_data, self.lat_lon_coords, self.time_stamps= self._load_variable_data(self.input_variables)

    def dates_to_tensor(self, date_list, fmt="%Y-%m-%d-%H-%M-%S"):
        # 1. 转换为 datetime 对象
        datetimes = [datetime.strptime(d, fmt) for d in date_list]
        # 2. 转换为 Unix 时间戳（秒）
        timestamps = [dt.timestamp() for dt in datetimes]
        # 3. 转换为 torch tensor
        return torch.tensor(timestamps, dtype=torch.float32)
    

    def _load_variable_data(self, vars, num_workers=8):
        """
        读取预处理后的npz文件，根据变量名vars和时间筛选进行加载。
        使用多线程加速文件读取和处理。
        """
        all_files = []
        for root_dir in self.root_dirs:
            if not os.path.exists(root_dir):
                raise FileNotFoundError(f"rootdir not found : {root_dir}")
            # 获取所有npy文件
            all_files += glob.glob(os.path.join(root_dir, '*.npy'))

        # 用于存储结果的列表
        all_data = []
        lat_lon_coords = []
        time_index = None
        
        # 定义一个处理单个文件的函数
        def process_file(file_path):
            nonlocal time_index
            try:
                # 加载npy文件
                npy_data = np.load(file_path, allow_pickle=True).item()  # 使用 `.item()` 获取字典

                data = npy_data['data']  # 变量数据
                dates = pd.to_datetime(npy_data['date'])  # 日期
                lat = npy_data['lat'].item()  # 纬度
                lon = npy_data['lon'].item()  # 经度
                var_names = npy_data['var_names'].tolist()  # 变量名列表
                if 't2m' in var_names:
                    var_names[0] = 'TMP'

                var_indices = [var_names.index(v) for v in vars]

                # 提取对应变量数据
                data = data[:, var_indices]

                # 时间筛选
                mask = (dates >= self.time_slice[0]) & (dates < self.time_slice[1])
                if not mask.any():
                    return None

                selected_data = data[mask][::self.step]
                selected_dates = dates[mask][::self.step]

                return {
                    'data': selected_data,
                    'dates': selected_dates,
                    'lat_lon': [lat, lon],
                    'file_path': file_path
                }
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
                return None

        # 使用线程池处理文件
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            # 使用tqdm显示进度
            results = list(tqdm(executor.map(process_file, all_files), 
                            total=len(all_files), 
                            desc="Loading .npy station data"))

        # 处理结果
        for result in results:
            if result is None:
                continue
                
            selected_data = result['data']
            selected_dates = result['dates']
            lat_lon = result['lat_lon']
            file_path = result['file_path']
            
            if time_index is None:
                time_index = selected_dates
            else:
                if not selected_dates.equals(time_index):
                    print("selected_dates mismatch: ", file_path)
                    continue

            all_data.append(selected_data)
            lat_lon_coords.append(lat_lon)

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
        output_tensors = output_tensors[:, :self.target_nodes, :]  # 只取第一个变量
        input_timestamps = self.time_stamps[idx:idx+self.time_window]
        # output_timestamps = self.time_stamps[idx+self.time_window:idx+self.time_window+self.pred_window]
        lat_lon = self.lat_lon_coords
        adj_tensor = self.adj
        # adj_tensor = torch.from_numpy(self.adj)

        return input_tensors, output_tensors, adj_tensor, input_timestamps, lat_lon


class GridDatasetAll(Dataset):
    def __init__(self,
                 weatherbench_dir,
                 process_npy_dir,
                 input_variables,
                 target_variables,
                 time_window,
                 pred_window,
                 time_slice=None,
                 step=1,
                 normalize=True):
        """
        初始化 CombinedDataset 数据集。

        参数:
        - weatherbench_dir (str): WeatherBench 数据集的根目录。
        - process_npy_dir (str): process_npy_data 生成的网格化数据目录。
        - input_variables (list): 输入变量的名称列表。
        - target_variables (list): 目标变量的名称列表。
        - time_window (int): 输入时间窗口的大小（时间步数）。
        - pred_window (int): 预测时间窗口的大小（时间步数）。
        - time_slice (tuple, optional): 时间范围 (start, end)。
        - step (int, optional): 时间步长，默认为1。
        """
        self.weatherbench_dir = weatherbench_dir
        self.process_npy_dir = process_npy_dir
        self.input_variables = input_variables
        self.target_variables = target_variables
        self.time_window = time_window
        self.pred_window = pred_window
        self.time_slice = time_slice
        self.step = step
        self.normalize = normalize

        # 加载 WeatherBench 数据
        self.weatherbench_data, self.weatherbench_times = self._load_weatherbench_data()

        # 加载 process_npy_data 数据
        self.process_npy_data, self.process_npy_times = self._load_process_npy_data()

        assert len(self.weatherbench_times) == len(self.process_npy_times)
        # 对齐时间戳
        self.time_stamps = self.process_npy_times


    def dates_to_tensor(self, date_list, fmt="%Y-%m-%d-%H-%M-%S"):
        # 1. 转换为 datetime 对象
        datetimes = [datetime.strptime(d, fmt) for d in date_list]
        # 2. 转换为 Unix 时间戳（秒）
        timestamps = [dt.timestamp() for dt in datetimes]
        # 3. 转换为 torch tensor
        return torch.tensor(timestamps, dtype=torch.float32)
    
    def _load_weatherbench_data(self):
        """
        加载指定变量的数据。

        参数:
        - var (str): 用户提供的变量名称。
        - mapped_var (str): 映射后的变量名称。
        - file_path (str): 文件路径模式。
        - data_list (list): 存储数据的列表。
        - channels_list (list): 存储通道数的列表。
        """

        if not os.path.exists(self.weatherbench_dir):
            raise FileNotFoundError(f"rootdir not found : {self.weatherbench_dir}")
        files = glob.glob(self.weatherbench_dir + '/*.nc')

        dataset = xr.open_mfdataset(files, combine="by_coords", engine="netcdf4")
        # dataset = xr.open_mfdataset(file_path, combine="by_coords")
        

        input_variables = VARIABLE_NAME_MAPPING[self.input_variables]
        # 读取变量数据
        var_data = dataset[input_variables]
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

        if self.normalize:
            mean = var_data.mean(dim=(2, 3), keepdim=True)
            std = var_data.mean(dim=(2, 3), keepdim=True)
            var_data = (var_data - mean) / std

        return var_data, time_stamps

    def _load_process_npy_data(self):
        """
        加载 process_npy_data 生成的数据。
        """
        files = glob.glob(os.path.join(self.process_npy_dir, "*.npy"))
        data_list = []
        time_list = []
        variable = 'temperature' if self.input_variables == '2m_temperature' else self.input_variables
        for file in files:
            data = np.load(file, allow_pickle=True).item()
            time_stamps = data['time']
            #time_stamps = [x.astype('datetime64[s]').astype(object).strftime('%Y-%m-%d-%H-%M-%S') for x in time_stamps]

            # 转换时间戳为 datetime 对象
            time_stamps_datetime = pd.to_datetime(time_stamps)

            # 根据时间范围筛选
            if self.time_slice:
                start_time = pd.to_datetime(self.time_slice[0])
                end_time = pd.to_datetime(self.time_slice[1]) + pd.Timedelta(days=1)  # 包含结束时间
                mask = (time_stamps_datetime >= start_time) & (time_stamps_datetime < end_time)
                if not mask.any():
                    continue  # 如果没有满足条件的时间，跳过该文件

                filtered_data = data[variable][mask]
                filtered_time_stamps = [time_stamps[i] for i in range(len(time_stamps)) if mask[i]]
                filtered_time_stamps = [x.astype('datetime64[s]').astype(object).strftime('%Y-%m-%d-%H-%M-%S') for x in filtered_time_stamps]
                data_list.append(filtered_data)
                time_list.extend(filtered_time_stamps)
            else:
                data_list.append(data[variable])
                time_list.extend(time_stamps)

        data = np.concatenate(data_list, axis=0)  # 在变量维度上拼接
        time_list = np.array(time_list)  # 在时间维度上拼接
        if len(data.shape) == 3:
            data = np.expand_dims(data, axis=1)  # 添加通道维度
        time_list = self.dates_to_tensor(time_list)
        return torch.tensor(data, dtype=torch.float32), time_list

    def __len__(self):
        return len(self.time_stamps) - self.time_window - self.pred_window + 1

    def __getitem__(self, idx):
        """
        获取指定索引的数据。

        参数:
        - idx (int): 索引。

        返回:
        - tuple: 输入张量、输出张量和时间戳张量。
        """


        input_weatherbench_grid = self.weatherbench_data[idx: idx + self.time_window]
        input_weather5k_grid = self.process_npy_data[idx: idx + self.time_window]

        input_tensors = torch.cat((input_weatherbench_grid, input_weather5k_grid), dim=1)

        output_tensors = self.weatherbench_data[idx + self.time_window: idx + self.time_window + self.pred_window]
        input_timestamps = self.time_stamps[idx: idx + self.time_window]


        return input_tensors, output_tensors, input_timestamps




if __name__ == "__main__":
    # weatherbench_dir = "/data/zyd/data/all_1.40625deg/2m_temperature/media/rasp/Elements/weather-benchmark/1.40625deg/2m_temperature"
    # process_npy_dir = "/data/zyd/data/weather_5k/WEATHER-5K/global_weather_stations_grid"
    # input_variables = "2m_temperature"  # 输入变量
    # target_variables = "2m_temperature" # 目标变量
    # time_window = 6
    # pred_window = 6
    # time_slice = ("2014-01-01 00:00:00", "2014-12-31 23:59:59")
    # step = 6

    # dataset = GridDatasetAll(
    #     weatherbench_dir=weatherbench_dir,
    #     process_npy_dir=process_npy_dir,
    #     input_variables=input_variables,
    #     target_variables=target_variables,
    #     time_window=time_window,
    #     pred_window=pred_window,
    #     time_slice=time_slice,
    #     step=step
    # )

    # print(f"Dataset size: {len(dataset)}")

    # # 测试数据加载
    # input_data, target_data, input_timestamps = dataset[0]
    # print("Input shape:", input_data.shape)
    # print("Target shape:", target_data.shape)




    # # Weather5k 数据集参数
    weather5k_params = {
        "root_dirs": ['/data/zyd/data/weather_5k/WEATHER-5K/global_weather_stations_npy_2','/data/zyd/data/all_1.40625deg/2m_temperature_graph_npy'],
        "input_variables": ["TMP"],
        "target_variables": ['TMP'],
        "time_window": 6,
        "pred_window": 6,
        "time_slice": ("2014-01-01", "2014-01-31"),
        "step": 1,
        "adj_path": '/data/zyd/data/weather_5k/WEATHER-5K/normalized_adj_2.npz',
    }

    # 初始化 Weather5k 数据集
    weather5k_dataset = GraphDatasetAll(**weather5k_params)
    train_loader = DataLoader(weather5k_dataset, batch_size=1, num_workers=2, shuffle=True)

    # 测试 Weather5k 数据加载器
    for i, data in enumerate(train_loader):
        print("input_tensors shape:", data[0].shape)  
        print("output_tensors shape:",  data[1].shape) 
        break