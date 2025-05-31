import torch
import torch.nn as nn
import torch.nn.functional as F

from graph_grid_fusion import *
from predictor import *
from ode import *
import time

class STG2CM(nn.Module):
    def __init__(self,
                h,
                w,
                patch_size,
                grid_input_dim,
                grid_feature_dim,
                grid_input_t,

                graph_nodes,
                graph_input_dim,
                graph_feature_dim,
                graph_input_t,
                
                target_dim_v,
                target_dim_t_grid,
                target_dim_t_graph):
        
        super(STG2CM, self).__init__()
        self.ode = NeuralODEModel(input_channels=grid_input_dim, hidden_dim=16, output_channels=graph_input_dim, T_prime=graph_input_t)

        self.graph2grid = Graph2Grid(h=h,
                                     w=w,
                                     graph_nodes=graph_nodes,
                                     grid_channels=grid_input_dim,
                                     graph_features=graph_input_dim,
                                     feature_dims=graph_feature_dim,
                                     ode_model=self.ode)
        

        self.grid2graph = Grid2Graph(patch_size=patch_size,
                                     grid_channels=grid_input_dim,
                                     graph_channel=graph_input_dim,
                                     graph_nodes=graph_nodes,
                                     node_features=graph_feature_dim)
        
        self.graph_grid_fusion = GraphGridCrossAttnModel(h,
                                                         w,
                                                         graph_feature_dim,
                                                         graph_feature_dim,
                                                         graph_feature_dim,
                                                         graph_feature_dim,
                                                         graph_input_t, num_heads=4)
        self.predictor = TimeSeriesPredictor(
                grid_input_channels=grid_feature_dim,
                grid_hidden_channels=grid_feature_dim,
                grid_output_channels=target_dim_v,
                graph_input_dim=graph_feature_dim,
                graph_hidden_dim=graph_feature_dim,
                graph_output_dim=target_dim_v,
                grid_t_in = grid_input_t,
                grid_t_out = target_dim_t_grid,
                graph_t_in = graph_input_t,
                graph_t_out = target_dim_t_graph
            )
        
    def forward(self, graph_data, grid_data, lat_lon_coords, graph_time_indices, grid_time_indices):
        time1 = time.time()
        grid_features = self.graph2grid(graph_data, grid_data, lat_lon_coords, graph_time_indices, grid_time_indices)
        time2 = time.time()
        # print("grid_features", grid_features)
        # print("graph2grid time cost: ", time2-time1)
        graph_features = self.grid2graph(grid_data, graph_data, lat_lon_coords, graph_time_indices, grid_time_indices)
        time3 = time.time()
        # print("graph_features", graph_features)
        # print("grid2graph time cost: ", time3-time2)
        graph_features, grid_features = self.graph_grid_fusion(graph_features, grid_features, graph_time_indices, grid_time_indices)
        time4 = time.time()
        # print("graph_features fusion", graph_features.shape)
        # print("grid_features fusion", grid_features.shape)
        # print("graph_grid_fusion time cost: ", time4-time3)
        grid_pred, graph_pred = self.predictor(grid_features, graph_features)
        
        # print(grid_pred.shape)
        # print(graph_pred.shape)
        time5 = time.time() 
        # print("predictor time cost: ", time5-time4)

        
        return grid_pred, graph_pred

if __name__ == "__main__":
    # Hyperparameters
    h, w = 16, 32  # Grid height and width
    T, T_prime = 5, 10  # Number of time steps for grid and graph
    C, C_prime = 3, 4  # Number of channels for grid and graph
    N = 100  # Number of graph nodes
    feature_dims = 32  # Node feature dimensionality
    input_dim = 1  # Input dimension for time embeddings
    patch_size = 8  # Patch size for grid2graph module
    batchsize = 2  # 假设批次大小为 2

    target_t = 6
    target_v = 1
    # Dummy inputs
    graph_data = torch.randn(batchsize, T_prime, N, C_prime)  # Batch size 2, T_prime time steps, N nodes, C_prime features per node
    grid_data = torch.randn(batchsize, T, C, h, w)  # Batch size 2, T time steps, C channels, h*w spatial resolution
    lat_lon_coords = torch.rand(batchsize, N, 2)  # Latitude and longitude coordinates for each graph node (normalized between 0 and 1)

    # 定义起始时间和结束时间
    start_time = datetime(2024, 1, 1, 0, 0, 0)  # 2024-01-01 00:00:00
    end_time = datetime(2024, 1, 1, 9, 0, 0)   # 2024-01-01 23:00:00

    # 生成多个批次的时间戳数据
    graph_time_indices = [generate_time_indices(start_time + timedelta(days=i), end_time + timedelta(days=i), 1) for i in range(batchsize)]
    grid_time_indices = [generate_time_indices(start_time + timedelta(days=i), end_time + timedelta(days=i), 2) for i in range(batchsize)]

    # 转换为 Unix 时间戳
    graph_time_indices = [to_unix(batch) for batch in graph_time_indices]
    grid_time_indices = [to_unix(batch) for batch in grid_time_indices]

    # 转换为 PyTorch 张量
    graph_time_indices = torch.tensor(graph_time_indices, dtype=torch.float32)  # (batchsize, T_prime)
    grid_time_indices = torch.tensor(grid_time_indices, dtype=torch.float32)    # (batchsize, T)

    model = STG2CM(h,
                w,
                patch_size,
                grid_input_dim = C,
                grid_feature_dim = feature_dims,
                grid_input_t = T,

                graph_nodes=N,
                graph_input_dim = C_prime,
                graph_feature_dim = feature_dims,
                graph_input_t = T_prime,
                
                target_dim_v = target_v,
                target_dim_t_grid = target_t,
                target_dim_t_graph = target_t)
    
    predicted_grid, predicted_graph = model(graph_data, grid_data, lat_lon_coords, graph_time_indices, grid_time_indices)
    
    # print(predicted_graph.shape)  # Expected shape: (batch_size, T, output_dim)
    # print(predicted_grid.shape)  # Expected shape: (batch_size, T, output_dim)
    
    from torchsummary import summary

    summary(model, input_size=[graph_data.shape, grid_data.shape, lat_lon_coords.shape, graph_time_indices.shape, grid_time_indices.shape])  # 输入尺寸需根据模型调整