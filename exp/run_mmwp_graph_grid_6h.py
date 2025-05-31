import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,Dataset,ConcatDataset
from tqdm import tqdm
import os
import xarray as xr
from dataset_st.weatherbench import WeatherBenchDataset
from dataset_st.weather5k import Weather5kDataset
from dataset_st.graph_grid_dataset import GraphDatasetAll,GridDatasetAll
from dataset_st.weather import GridGraphDataset

from train import Trainer  
from comet_ml import Experiment
from loguru import logger
from models.mmwp import MMWP
import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.append('/data/zyd/codes/spatial-temporal/models/graph_grid_model/graph_grid_fusion.py')

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# sys.path.append('/home/jovyan/work/spatial-temporal/models/graph_grid_model')
logger.add("./logs/training_log.log", rotation="1 MB", level="INFO")

# output_model_path = '/data/zyd/codes/spatial-temporal/outputs/0510_96_96/models'
output_model_path = '/data/zyd/codes/spatial-temporal/outputs/0511_6h/models'
os.makedirs(output_model_path, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# time_slice_train_grid = ("2018-01-01", "2018-01-31")  # 加载指定时间范围的数据
# time_slice_val_grid = ('2018-02-01', '2018-02-10') # 加载指定时间范围的数据
# time_slice_test_grid = ('2018-02-11', '2018-02-20')  # 加载指定时间范围的数据

# time_slice_train_graph = ("2018-01-01", "2018-02-01")  # 加载指定时间范围的数据
# time_slice_val_graph = ('2018-02-01', '2018-02-11') # 加载指定时间范围的数据
# time_slice_test_graph = ('2018-02-11', '2018-02-21')  # 加载指定时间范围的数据

time_slice_train_grid = ("2014-01-01", "2016-12-31")  # 加载指定时间范围的数据
time_slice_val_grid = ('2017-01-01', '2017-12-31') # 加载指定时间范围的数据
time_slice_test_grid = ('2018-01-01', '2018-12-31')  # 加载指定时间范围的数据

time_slice_train_graph = ("2014-01-01", "2017-01-01")  # 加载指定时间范围的数据
time_slice_val_graph = ('2017-01-01', '2018-01-01') # 加载指定时间范围的数据
time_slice_test_graph = ('2018-01-01', '2019-01-01')  # 加载指定时间范围的数据

in_channels = 1
hidden_channels = 32
out_channels = 1
in_t = 8
out_t = 1
input_nodes = 38438
target_nodes = 5671

batch_size = 2
learning_rate = 1e-4
num_epochs = 50


weather5k_params = {
    # "root_dir": '/data/zyd/data/weather_5k/WEATHER-5K/global_weather_stations_test',
    "root_dirs": ['/data/zyd/data/weather_5k/WEATHER-5K/global_weather_stations_npy_2','/data/zyd/data/all_1.40625deg/2m_temperature_graph_npy_2'],
    "input_variables": ["TMP"],
    "target_variables": ['TMP'],
    "time_window": in_t,
    "pred_window": out_t,
    "step": 6,
    "adj_path": '/data/zyd/data/weather_5k/WEATHER-5K/normalized_adj_2.npz',
    "target_nodes": target_nodes,
}

graph_dataset_train = GraphDatasetAll(time_slice=time_slice_train_graph,**weather5k_params)
graph_dataset_val = GraphDatasetAll(time_slice=time_slice_val_graph,**weather5k_params)
graph_dataset_test = GraphDatasetAll(time_slice=time_slice_test_graph,**weather5k_params)



weatherbench_params = {
    "weatherbench_dir": "/data/zyd/data/all_1.40625deg/2m_temperature/media/rasp/Elements/weather-benchmark/1.40625deg/2m_temperature",
    "process_npy_dir" : "/data/zyd/data/weather_5k/WEATHER-5K/global_weather_stations_grid",
    "input_variables": "2m_temperature",  
    "target_variables": "2m_temperature", 
    "time_window": in_t,
    "pred_window": out_t,
    "step":6
}

grid_dataset_train = GridDatasetAll(time_slice=time_slice_train_grid, **weatherbench_params)
grid_dataset_val = GridDatasetAll(time_slice=time_slice_val_grid, **weatherbench_params)
grid_dataset_test = GridDatasetAll(time_slice=time_slice_test_grid, **weatherbench_params)


weather_dataset_train = GridGraphDataset(grid_dataset_train, graph_dataset_train)
weather_dataset_val = GridGraphDataset(grid_dataset_val, graph_dataset_val)
weather_dataset_test = GridGraphDataset(grid_dataset_test, graph_dataset_test)


train_loader = DataLoader(weather_dataset_train, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(weather_dataset_val, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(weather_dataset_test, batch_size=1, shuffle=False)


model = MMWP(in_channels = in_channels,
                hidden_channels = hidden_channels,
                out_channels = out_channels, 
                in_t = in_t,
                out_t = out_t,
                input_nodes = input_nodes,
                target_nodes = target_nodes)

model.to(device)

# 损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#scheduler = CyclicLR(optimizer, base_lr=learning_rate, max_lr=1e-3, cycle_momentum=False)
experiment = Experiment(
    api_key="xxx",  # 替换为你的 Comet API Key
    project_name="spatialtemporal",
    workspace="yyy"  # 替换为你的 Comet 工作区
)


##### Comet.ml 实验记录
experiment.log_parameters({
    "batch_size": batch_size,
    "num_epochs": num_epochs,
    "learning_rate": learning_rate
})

# 训练器实例化
trainer = Trainer(
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    device=device,
    output_path=output_model_path,
    experiment=experiment,
    predict_path='/data/zyd/codes/spatial-temporal/outputs/0511_6h'
)

# 训练循环
for epoch in range(num_epochs):
    logger.info(f"Epoch [{epoch+1}/{num_epochs}]")
    train_loss = trainer.train_epoch_grid_graph()
    val_metrics = trainer.validate_epoch_grid_graph()

# 测试
# trainer.precict_grid_graph()


# 结束实验
experiment.end()
