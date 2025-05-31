import torch
import torch.nn as nn
import torch.nn.functional as F

# 引入你之前写的 SimpleGCN
class SimpleGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleGCN, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        self.act = nn.ReLU()

        nn.init.xavier_normal_(self.linear1.weight)
        nn.init.zeros_(self.linear1.bias)
        nn.init.xavier_normal_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)

    def forward(self, x, edge_index, edge_weights):
        num_nodes = x.size(0)

        # 添加自环
        loop_idx = torch.arange(0, num_nodes, device=x.device)
        self_loop = torch.stack([loop_idx, loop_idx], dim=0)
        edge_index = torch.cat([edge_index, self_loop], dim=1)

        self_loop_weight = torch.full((num_nodes,), 0.5, device=x.device)
        edge_weights = torch.cat([edge_weights, self_loop_weight], dim=0)

        x_transformed = self.linear1(x)
        x_transformed = self.act(x_transformed)

        source_nodes = edge_index[0]
        target_nodes = edge_index[1]
        messages = x_transformed[source_nodes] * edge_weights.unsqueeze(-1)

        aggregated = torch.zeros_like(x_transformed)
        aggregated = aggregated.index_add(0, target_nodes, messages)

        out = self.linear2(x_transformed + aggregated)
        return out

# ----------- 测试数据构造（含边权重 = 距离倒数） ---------------

class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleMLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        self.act = nn.ReLU()
        
        # 初始化参数
        nn.init.xavier_normal_(self.linear1.weight)
        nn.init.zeros_(self.linear1.bias)
        nn.init.xavier_normal_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)

    def forward(self, x):
        # edge_index 和 edge_weights 不再使用，仅保留参数接口
        x = self.linear1(x)
        x = self.act(x)
        x = self.linear2(x)
        return x

if __name__ == "__main__":

    num_nodes = 10
    input_dim = 5
    hidden_dim = 16
    output_dim = 3

    # 1. 节点特征
    x = torch.randn(num_nodes, input_dim)

    # 2. 构造完全图的边（不含自环）
    row, col = torch.meshgrid(torch.arange(num_nodes), torch.arange(num_nodes), indexing='ij')
    mask = row != col  # 去掉自环
    edge_index = torch.stack([row[mask], col[mask]], dim=0)  # shape: (2, N*(N-1))

    # 3. 计算距离并取倒数作为边权重
    x_i = x[edge_index[0]]  # source
    x_j = x[edge_index[1]]  # target
    distance = torch.norm(x_i - x_j, p=2, dim=1)  # 欧几里得距离
    edge_weights = 1.0 / (distance + 1e-8)  # 防止除0

    # 4. 初始化模型并前向
    model = SimpleGCN(input_dim, hidden_dim, output_dim)
    output = model(x, edge_index, edge_weights)

    print("输出 shape:", output.shape)