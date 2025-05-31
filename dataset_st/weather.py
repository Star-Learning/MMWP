import os
import sys
sys.path.append('D:\work\codes\proj\spatial-temporal')
import torch
from torch.utils.data import Dataset, DataLoader
# from weather5k import Weather5kDataset
# from weatherbench import WeatherBenchDataset

class GridGraphDataset(Dataset):
    def __init__(self, grid_dataset, graph_dataset):
        """
        Weather-5K Dataset for time series prediction.
        """
        self.grid_dataset = grid_dataset
        self.graph_dataset = graph_dataset

        ## 两者长度一致，时间对应上
        print('grid_dataset', grid_dataset.__len__())
        print('graph_dataset', graph_dataset.__len__())
        assert self.grid_dataset.__len__() == self.graph_dataset.__len__()

    def __len__(self):
        return self.grid_dataset.__len__()

    def __getitem__(self, idx):
        data_grid = self.grid_dataset[idx]
        data_graph = self.graph_dataset[idx]

        return data_grid, data_graph
    
if __name__ == "__main__":
    pass
    weatherbench_params = {
        "root_dir": r'D:\work\data\2m_temperature',
        "input_variables": ["2m_temperature"],
        "target_variables": ["2m_temperature"],
        "time_window": 6,
        "pred_window": 6,
        "time_slice": ("2014-01-01", "2014-01-10"),
        "step": 1,
    }
    # weatherbench_dataset = WeatherBenchDataset(**weatherbench_params)

    weather5k_params = {
        "root_dir": 'D:/work/data/WEATHER-5K/global_weather_stations_test',
        "input_variables": ["TMP"],
        "target_variables": ['TMP'],
        "time_window": 6,
        "pred_window": 6,
        "time_slice": ("2014-01-01", "2014-01-11"),
        "step": 6,
    }

    # weather5k_dataset = Weather5kDataset(**weather5k_params)
    # weather_dataset = GridGraphDataset(weatherbench_dataset, weather5k_dataset)

    # train_loader = DataLoader(weather_dataset, batch_size=1, num_workers=2, shuffle=True)

    # # 测试 Weather5k 数据加载器
    # for i, (data_grid, data_graph) in enumerate(train_loader):
    #     print("data_grid", data_grid)  
    #     print("data_graph", data_graph)  
    #     break
