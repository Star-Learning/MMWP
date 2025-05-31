import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.data import Data, Batch
import torch_geometric.utils as utils
from torch_geometric.utils import add_self_loops
from torch_scatter import scatter_softmax  # Added for efficient edge weight normalization


class SimpleGNN(MessagePassing):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__(aggr="add")
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        self.act = nn.ReLU()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # 初始化参数
        nn.init.xavier_normal_(self.linear1.weight)
        nn.init.zeros_(self.linear1.bias)
        nn.init.xavier_normal_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)

    def forward(self, x, edge_index, edge_weights):
        # 检查输入 x 的形状
        # print(f"x.shape: {x.shape}, edge_index.shape: {edge_index.shape}")

        # 添加断言，确保 edge_index 的值合法
        assert edge_index.max().item() < x.size(0), f"edge_index contains invalid indices: max={edge_index.max().item()}, x.size(0)={x.size(0)}"
        assert edge_index.min().item() >= 0, f"edge_index contains negative indices: min={edge_index.min().item()}"

        # 添加自环边（自定义权重）
        edge_index, edge_weights = add_self_loops(
            edge_index, edge_weights, 
            fill_value=0.5,  # Directly use scalar instead of creating a tensor
            num_nodes=x.size(0)
        )
        
        # 特征变换
        x = self.linear1(x)
        x = self.act(x)
        
        # 消息聚合
        aggregated = self.propagate(edge_index, x=x, edge_weights=edge_weights)
        
        # 残差连接 + 输出
        x = x + aggregated
        x = self.linear2(x)
        return x

    def message(self, x_j, edge_weights, index, ptr, size_i):
        # 边权重归一化 (optimized with scatter_softmax)
        edge_weights = scatter_softmax(edge_weights, index, dim=0)
        return edge_weights.view(-1, 1) * x_j
    
# 示例数据
# num_nodes = 10000
# input_dim = 3
# hidden_dim = 64
# output_dim = 128
# x = torch.randn(num_nodes, input_dim)
# edge_index = torch.randint(0, num_nodes, (2, num_nodes * 10))

# # 去重并排序边索引
# edge_index = utils.coalesce(edge_index, num_nodes=num_nodes)

# # 模型实例化
# model = SimpleGNN(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)

# # 前向传播
# output_features = model(x, edge_index)
# print(output_features.shape)  # 输出应为 (num_nodes, output_dim)