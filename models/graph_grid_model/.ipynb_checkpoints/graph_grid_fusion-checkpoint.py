import torch
import torch.nn as nn
import torch.nn.functional as F

from datetime import datetime, timedelta
from einops import rearrange, reduce

from gnn import *
from ode import *

class Graph2Grid(nn.Module):
    def __init__(self, h, w, graph_nodes, grid_channels, graph_features, feature_dims, ode_model):
        super(Graph2Grid, self).__init__()
        self.h, self.w = h, w
        self.linear_graph = nn.Linear(graph_features, feature_dims // 2)
        self.linear_grid = nn.Linear(grid_channels, feature_dims // 2)
        self.ode_model = ode_model
        self.norm = nn.BatchNorm3d(feature_dims)  # 使用 BatchNorm3d 替代 LayerNorm

    def forward(self, graph_data, grid_data, lat_lon_coords, graph_time_indices, grid_time_indices):
        batch_size, T_prime, N, C_prime = graph_data.shape
        _, T, C, H, W = grid_data.shape

        print("grid_data.shape:", grid_data.shape)
        grid_data_ode = self.ode_model(grid_data).permute(0, 1, 3, 4, 2)
        print("grid_data_ode.shape:", grid_data_ode.shape)

        interp_grid = self.interpolate_to_grid(graph_data, lat_lon_coords, H, W).permute(0, 1, 3, 4, 2)

        interp_grid = self.linear_graph(interp_grid).permute(0, 1, 4, 2, 3)
        grid_data_ode = self.linear_grid(grid_data_ode).permute(0, 1, 4, 2, 3)

        combined_grid = torch.cat([interp_grid, grid_data_ode], dim=2)  # Shape: [batch_size, T, feature_dims, H, W]

        # 调整维度以适配 BatchNorm3d
        combined_grid = combined_grid.permute(0, 2, 1, 3, 4)  # Shape: [batch_size, feature_dims, T, H, W]
        combined_grid = self.norm(combined_grid)  # 归一化
        combined_grid = combined_grid.permute(0, 2, 1, 3, 4)  # 调整回原始维度顺序


        print("combined_grid.shape:", combined_grid.shape)
        return combined_grid

    def interpolate_to_grid(self, graph_data, lat_lon_coords, H, W):
        batch_size, T, N, C = graph_data.shape
        grid_maps = torch.zeros(batch_size, T, C, H, W, device=graph_data.device)

        lat_grid = torch.clamp((lat_lon_coords[..., 0] * H).long(), 0, H - 1)
        lon_grid = torch.clamp((lat_lon_coords[..., 1] * W).long(), 0, W - 1)

        for t in range(T):
            for i in range(batch_size):
                grid_maps[i, t].index_put_(
                    (torch.arange(C).unsqueeze(1), lat_grid[i], lon_grid[i]),
                    graph_data[i, t].T,
                    accumulate=True
                )
        return grid_maps


class Grid2Graph(nn.Module):
    def __init__(self, patch_size, grid_channels, graph_channel, graph_nodes, node_features):
        super(Grid2Graph, self).__init__()
        self.patch_size = patch_size
        self.patch_embed = nn.Conv2d(grid_channels, node_features, kernel_size=patch_size, stride=patch_size)
        self.node_embed = nn.Linear(1, node_features)
        self.gnn = SimpleGNN(input_dim=node_features, hidden_dim=node_features * 2, output_dim=node_features)
        self.norm = nn.LayerNorm(node_features)  # 增加归一化层
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 定义设备属性

    def forward(self, grid_data, graph_data, lat_lon_coords, graph_time_indices, grid_time_indices):
        batch_size, T, C, H, W = grid_data.shape
        _, T_prime, N, C_prime = graph_data.shape
        all_node_features = []

        for b in range(batch_size):
            batch_node_features = []
            for t_idx in range(graph_time_indices.size(1)):
                grid_idx = t_idx

                # 对 grid 数据进行 patch_embed
                patches = self.patch_embed(grid_data[b, grid_idx]).view(1, -1, self.gnn.input_dim)  # Shape: [1, num_patches, feature_dim]
                patch_loc = self.get_grid_loc(H // self.patch_size, W // self.patch_size, 1).to(self.device)  # 获取 grid 的相对坐标

                # 对 graph 数据进行处理
                
#                 print(patch_loc.shape)
#                 print(patch_loc)
                
                graph_features = self.node_embed(graph_data[b, t_idx]).unsqueeze(0)  # Shape: [1, num_graph_nodes, feature_dim]
                graph_loc = self.get_graph_loc(lat_lon_coords[b:b+1], 1).to(self.device)  # 获取 graph 的归一化经纬度坐标
                # print(graph_loc.shape)
                # print(graph_loc)
                # 拼接 grid 的 patch 和 graph 的节点
                combined_features = torch.cat([graph_features, patches], dim=1)  # Shape: [1, num_combined_nodes, feature_dim]
                combined_loc = torch.cat([graph_loc, patch_loc], dim=1)  # Shape: [1, num_combined_nodes, 2]

                # 检查拼接后的节点数量是否一致
                assert combined_features.size(1) == combined_loc.size(1), (
                    f"Mismatch in combined_features and combined_loc: "
                    f"combined_features.size(1)={combined_features.size(1)}, combined_loc.size(1)={combined_loc.size(1)}"
                )

                # 在拼接后的节点之间计算边关系
                edge_index, edge_weights = self.compute_weight_matrix(combined_loc)

                # 限制 edge_index 的范围
                edge_index = edge_index[:, edge_index[1] < combined_features.size(1)]

                # 调试信息
                # print(f"combined_features.shape: {combined_features.shape}")
                # print(f"edge_index.shape: {edge_index.shape}, edge_index.max: {edge_index.max().item()}, edge_index.min: {edge_index.min().item()}")

                # 通过图神经网络进行特征变换
                x = self.gnn(combined_features.squeeze(0), edge_index, edge_weights)  # Shape: [num_combined_nodes, feature_dim]

                # 只保留 graph 节点的特征
                batch_node_features.append(x[:N])
            # print(torch.stack(batch_node_features, dim=0).shape)
            all_node_features.append(torch.stack(batch_node_features, dim=0))
        result = self.norm(torch.stack(all_node_features, dim=0))  # 特征归一化
        # print(result.shape)
        return result

    def get_grid_loc(self, H, W, batch_size):
        """
        获取 grid 的相对坐标。

        参数:
        - H: grid 的高度
        - W: grid 的宽度
        - batch_size: 批次大小

        返回:
        - grid_coords: 形状为 (batch_size, H * W, 2) 的相对坐标
        """
        # 生成 [0, 1] 范围内的相对坐标
        x = torch.linspace(0, 1, W, device=self.device)
        y = torch.linspace(0, 1, H, device=self.device)
        grid_coords = torch.stack(torch.meshgrid(y, x), dim=-1).view(-1, 2)  # Shape: (H * W, 2)

        # 扩展到 batch_size
        grid_coords = grid_coords.unsqueeze(0).expand(batch_size, -1, -1)  # Shape: (batch_size, H * W, 2)
        return grid_coords

    def get_graph_loc(self, coords, batch_size):
        """
        获取 graph 的归一化经纬度坐标。

        参数:
        - coords: 原始经纬度坐标，形状为 (batch_size, N, 2)
        - batch_size: 批次大小

        返回:
        - normalized_coords: 形状为 (batch_size, N, 2) 的归一化坐标
        """
        # 检查输入维度
        assert coords.dim() == 3 and coords.size(-1) == 2, (
            f"coords must have shape (batch_size, N, 2), but got {coords.shape}"
        )

        # 归一化经纬度坐标
        normalized_coords = torch.stack([
            (coords[..., 0] + 90) / 180,  # 纬度归一化到 [0, 1]
            (coords[..., 1] + 180) / 360  # 经度归一化到 [0, 1]
        ], dim=-1)  # Shape: (batch_size, N, 2)

        return normalized_coords

    def compute_weight_matrix(self, combined_loc, epsilon=1e-10, k=10):
        """
        计算拼接后的节点之间的权重矩阵。

        参数:
        - combined_loc: 形状为 (batch_size, N, 2) 的拼接节点坐标
        - epsilon: 防止除以零的常数
        - k: 每个节点保留最近的 k 个邻居

        返回:
        - edge_index: 边索引，形状为 (2, num_edges)
        - edge_weights: 边权重，形状为 (num_edges,)
        """
        device = combined_loc.device
        num_nodes = combined_loc.size(1)

        # 计算拼接节点之间的距离
        diff = combined_loc.unsqueeze(2) - combined_loc.unsqueeze(1)  # Shape: (batch_size, N, N, 2)
        dist = torch.norm(diff, p=2, dim=-1)  # Shape: (batch_size, N, N)

        # 对每个节点保留最近的 k 个邻居
        topk_values, topk_indices = torch.topk(dist, k=min(k, num_nodes), dim=-1, largest=False)  # Shape: (batch_size, N, k)
        weights = 1 / (topk_values + epsilon)  # Shape: (batch_size, N, k)

        # 构造稀疏边索引
        src_indices = torch.arange(num_nodes, device=device).view(1, -1, 1).expand_as(topk_indices)  # Shape: (batch_size, N, k)
        edge_index = torch.stack([src_indices.flatten(), topk_indices.flatten()], dim=0)  # Shape: (2, num_edges)
        edge_weights = weights.flatten().to(device)  # Shape: (num_edges,)

        return edge_index, edge_weights


class CrossAttention(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads=4):
        super(CrossAttention, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads
        
        assert self.head_dim * num_heads == output_dim, "output_dim must be divisible by num_heads"
        
        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(input_dim, output_dim)
        self.k_proj = nn.Linear(input_dim, output_dim)
        self.v_proj = nn.Linear(input_dim, output_dim)
        
        # Final output projection
        self.out_proj = nn.Linear(output_dim, output_dim)
        
    def forward(self, Q, K, V):
        batch_size, q_len, _ = Q.size()
        _, k_len, _ = K.size()
        
        # Project Q, K, V
        Q = self.q_proj(Q)  # [batch_size, q_len, output_dim]
        K = self.k_proj(K)  # [batch_size, k_len, output_dim]
        V = self.v_proj(V)  # [batch_size, k_len, output_dim]
        
        # Reshape Q, K, V for multi-head attention
        Q = Q.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)  # [batch_size, num_heads, q_len, head_dim]
        K = K.view(batch_size, k_len, self.num_heads, self.head_dim).transpose(1, 2)  # [batch_size, num_heads, k_len, head_dim]
        V = V.view(batch_size, k_len, self.num_heads, self.head_dim).transpose(1, 2)  # [batch_size, num_heads, k_len, head_dim]
        
        # Scaled Dot-Product Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))  # [batch_size, num_heads, q_len, k_len]
        attn_weights = F.softmax(scores, dim=-1)  # [batch_size, num_heads, q_len, k_len]
        
        # Apply attention weights to V
        output = torch.matmul(attn_weights, V)  # [batch_size, num_heads, q_len, head_dim]
        
        # Reshape output back to [batch_size, q_len, output_dim]
        output = output.transpose(1, 2).contiguous().view(batch_size, q_len, self.output_dim)  # [batch_size, q_len, output_dim]
        
        # Project output
        output = self.out_proj(output)  # [batch_size, q_len, output_dim]
        
        # Interpolate output to match k_len
        output = output.permute(0, 2, 1)  # [batch_size, output_dim, q_len]
        output = F.interpolate(output, size=k_len, mode='linear', align_corners=False)  # [batch_size, output_dim, k_len]
        output = output.permute(0, 2, 1)  # [batch_size, k_len, output_dim]
        
        return output


    
class PatchEmbedding(nn.Module):
    def __init__(self, d, patch_size, embed_dim):
        """
        初始化 Patch Embedding 模块。
        :param d: 输入通道数
        :param patch_size: patch 的大小 (patch_h, patch_w)
        :param embed_dim: 嵌入的维度
        """
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size  # (patch_h, patch_w)
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(d, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        """
        :param x: 输入张量，形状为 (batch_size, t, d, h, w)
        :return: patch embedding，形状为 (batch_size, t, n_patches, embed_dim)
        """
        batch_size, t, d, h, w = x.shape
        patch_h = patch_w = self.patch_size

        # 调整形状，将每个时间步当作一个独立图像
        # x = x.view(batch_size * t, d, h, w)  # (batch_size * t, d, h, w)
        x = rearrange(x, 'b t d h w -> (b t) d h w')
        # 使用 Conv2D 投影
        x = self.proj(x)  # (batch_size * t, embed_dim, h_p, w_p)

        # 获取 patch 划分后的高宽
        h_p, w_p = x.shape[-2], x.shape[-1]
        n_patches = h_p * w_p

        # 调整输出形状
        x = x.flatten(2)  # (batch_size * t, embed_dim, n_patches)
        x = x.transpose(1, 2)  # (batch_size * t, n_patches, embed_dim)
        x = x.view(batch_size, t, n_patches, self.embed_dim)  # (batch_size, t, n_patches, embed_dim)

        return x


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        """
        :param in_features: 输入特征维度
        :param hidden_features: 隐藏层特征维度
        :param out_features: 输出特征维度
        """
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.activation = nn.ReLU()  # 激活函数
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x


class GraphGridCrossAttnModel(nn.Module):
    def __init__(self, h, w, graph_input_dim, grid_input_dim, embed_dim, pred_dim, t_prime, num_heads = 4,patch_size=8):
        super(GraphGridCrossAttnModel, self).__init__()
        self.h = h
        self.w = w
        self.graph_embed_k = MLP(graph_input_dim, embed_dim, embed_dim)
        self.graph_embed_v = MLP(graph_input_dim, embed_dim, embed_dim)

        self.grid_embed_k = MLP(grid_input_dim, embed_dim, embed_dim)
        self.grid_embed_v = MLP(grid_input_dim, embed_dim, embed_dim)

        self.graph_time_emb = MLP(1, embed_dim, embed_dim)
        self.grid_time_emb = MLP(1, embed_dim, embed_dim)

        #self.patch_size = (4,8)
        self.patch_size = patch_size
        self.patch_emb = PatchEmbedding(embed_dim, self.patch_size, embed_dim)
        self.upsample = nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)

        self.cross_attn_graph = CrossAttention(embed_dim,embed_dim, num_heads)
        self.cross_attn_grid = CrossAttention(embed_dim,embed_dim, num_heads)
        
        self.graph_output = nn.Linear(embed_dim, pred_dim)
        self.grid_output = nn.Conv2d(embed_dim, pred_dim, kernel_size=1)

        self.t_prime = t_prime


    def forward(self, graph_features, grid_features, graph_time_indices, grid_time_indices):
        batch_size, t, n, d = graph_features.shape
        _, _, _, h, w = grid_features.shape
        graph_features = graph_features.to(torch.float32)
        grid_features = grid_features.to(torch.float32)
        graph_time_indices = graph_time_indices.to(torch.float32)
        grid_time_indices = grid_time_indices.to(torch.float32)
        
        if len(graph_time_indices.shape) == 2:
            graph_time_indices = graph_time_indices.unsqueeze(-1)
        if len(grid_time_indices.shape) == 2:
            grid_time_indices = grid_time_indices.unsqueeze(-1)


        graph_time_indices = graph_time_indices[:,:,0].unsqueeze(-1)
        # Embedding graph time sequence
        graph_time_embedded = self.graph_time_emb(graph_time_indices)  # Shape: (batch_size, T, embed_dim)

        graph_k = self.graph_embed_k(graph_features.reshape(batch_size, t * n, d))
        graph_v = self.graph_embed_v(graph_features.reshape(batch_size, t * n, d))
        # print(graph_time_embedded.shape)
        # print(graph_k.shape)
        # print(graph_v.shape)
        graph_attn_output = self.cross_attn_graph(graph_time_embedded, graph_k, graph_v)  # (batchsize, t *, d)]

        print("graph_attn_output shape ",graph_attn_output.shape)
        graph_attn_output = rearrange(graph_attn_output, 'b (t n) c -> b t n c',t=t,n=n)

        # Grid data: (batchsize, t, c, h, w) -> (batchsize, t, h*w*c)
        # print(grid_features.shape)
        # grid_features = rearrange(grid_features, 'b t c h w -> b t (c h w)')
        grid_features_patched = self.patch_emb(grid_features)
        # print(grid_features_patched.shape)
        grid_k = rearrange(self.grid_embed_k(grid_features_patched), 'b t n c -> b (t n) c')
        grid_v = rearrange(self.grid_embed_v(grid_features_patched), 'b t n c -> b (t n) c')

        
        # print(graph_time_embedded.shape)
        # print(grid_k.shape)
        # print(grid_v.shape)
        grid_attn_output = self.cross_attn_grid(graph_time_embedded, grid_k, grid_v)  # (batchsize, t, d)
        # print(grid_attn_output.shape)

        grid_attn_output = rearrange(grid_attn_output, 'b (t h w) c -> (b t) c h w', t=t, h=self.h//self.patch_size, w=self.w//self.patch_size)
        grid_attn_output = self.upsample(grid_attn_output)
        grid_attn_output = rearrange(grid_attn_output, '(b t) c h w -> b t c h w', t=t)

        return graph_attn_output, grid_attn_output


class TimeSeriesPredictor(nn.Module):
    def __init__(self, grid_input_dim, graph_input_dim, target_variable, target_time, n):
        super(TimeSeriesPredictor, self).__init__()

        self.target_variable = target_variable
        self.target_time = target_time
        self.n = n
        self.grid_v_decoder = nn.Conv3d(grid_input_dim, target_variable, kernel_size=(3, 3, 3), padding=1)
        self.graph_v_decoder = MLP(graph_input_dim, graph_input_dim, target_variable)

    def forward(self, grid_input, graph_input):
        batch_size, t, d, h, w = grid_input.shape
        _, t_, n, d_ = graph_input.shape
        
        # Grid Decoder - Process with Conv3D
        grid_out = self.grid_v_decoder(grid_input)  # Output shape: (batch_size, target_variable, h, w)
        
        # Adjust time dimension using interpolation (e.g., linear interpolation)
        grid_out = F.interpolate(grid_out, size=(self.target_time, h, w), mode='trilinear', align_corners=False)
        grid_out = grid_out.permute(0, 2, 3, 4, 1)  # Shape: (batch_size, target_time, h, w, target_variable)
        
        # Graph Decoder - Process with MLP
        graph_out = self.graph_v_decoder(graph_input)  # Output shape: (batch_size, target_time, n, target_variable)
        
        return grid_out, graph_out




# 生成时间戳数据 
def generate_time_indices(start_time, end_time, interval_hours):
    return [(start_time + timedelta(hours=i)).strftime("%Y%m%d %H:%M:%S") for i in range(0, (end_time - start_time).seconds // 3600 + 1, interval_hours)]

# 转换为 Unix 时间戳
def to_unix(timestamps):
    return [int(datetime.strptime(ts, "%Y%m%d %H:%M:%S").timestamp()) for ts in timestamps]

# Example usage
if __name__ == "__main__":
    # Hyperparameters
    h, w = 64, 128  # Grid height and width
    T, T_prime = 5, 10  # Number of time steps for grid and graph
    C, C_prime = 3, 4  # Number of channels for grid and graph
    N = 100  # Number of graph nodes
    feature_dims = 64  # Node feature dimensionality
    input_dim = 20  # Input dimension for time embeddings
    hidden_dim = 64  # Hidden dimension for cross attention
    output_dim = 64  # Output dimension for final projections
    num_heads = 4  # Number of heads for multi-head attention
    patch_size = 8  # Patch size for grid2graph module
    batchsize = 2  # 假设批次大小为 2

    # Dummy inputs
    graph_data = torch.randn(batchsize, T_prime, N, C_prime)  # Batch size 2, T_prime time steps, N nodes, C_prime features per node
    grid_data = torch.randn(batchsize, T, C, h, w)  # Batch size 2, T time steps, C channels, h*w spatial resolution
    lat_lon_coords = torch.rand(batchsize, N, 2)  # Latitude and longitude coordinates for each graph node (normalized between 0 and 1)
    # graph_time_indices = list(range(T_prime))  # Indices mapping graph time steps to grid time steps
    # grid_time_indices = list(range(0, T_prime, 2))


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


    ode_model = NeuralODEModel(input_channels=C, hidden_dim=16, output_channels=C, T_prime=T_prime)

    g2g = Graph2Grid(h=h, w=w, graph_nodes=N, grid_channels=C, graph_features=C_prime,feature_dims=feature_dims, ode_model=ode_model)
    g2gr = Grid2Graph(patch_size=patch_size, grid_channels=C,graph_channel=C_prime, graph_nodes=N, node_features=feature_dims)
    ufm = GraphGridCrossAttnModel(h,w,output_dim, output_dim, hidden_dim, output_dim, T_prime, num_heads=num_heads)

    grid_features = g2g(graph_data, grid_data, lat_lon_coords, graph_time_indices, grid_time_indices)
    graph_features = g2gr(grid_data, graph_data, lat_lon_coords, graph_time_indices, grid_time_indices)
    predicted_graph, predicted_grid = ufm(graph_features, grid_features, graph_time_indices, grid_time_indices)
    
    # print(predicted_graph.shape)  # Expected shape: (batch_size, T, output_dim)
    # print(predicted_grid.shape)  # Expected shape: (batch_size, T, output_dim)