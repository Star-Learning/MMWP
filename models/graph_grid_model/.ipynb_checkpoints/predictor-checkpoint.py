import torch
import torch.nn as nn
import torch.nn.functional as F

class GridEncoder(nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels):
        super(GridEncoder, self).__init__()
        self.conv1 = nn.Conv3d(input_channels, hidden_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv2 = nn.Conv3d(hidden_channels, output_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1))

    def forward(self, x):
        # x: (batchsize, t, d, h, w)
        x = x.permute(0, 2, 1, 3, 4)  # (batchsize, d, t, h, w)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x  # (batchsize, output_channels, t, h, w)

class GraphEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GraphEncoder, self).__init__()
        self.mlp1 = nn.Linear(input_dim, hidden_dim)
        self.mlp2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: (batchsize, t, n, d)
        batchsize, t, n, d = x.shape
        x = x.reshape(batchsize * t, n, d)  # (batchsize * t, n, d)
        x = F.relu(self.mlp1(x))
        x = F.relu(self.mlp2(x))
        x = x.reshape(batchsize, t, n, -1)  # (batchsize, t, n, output_dim)
        return x

class GridDecoder(nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels, t_prime):
        super(GridDecoder, self).__init__()
        self.t_prime = t_prime
        self.conv1 = nn.Conv3d(input_channels, hidden_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv2 = nn.Conv3d(hidden_channels, output_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.upsample = nn.ConvTranspose3d(output_channels, output_channels, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=1)

    def forward(self, x):
        # x: (batchsize, output_channels, t, h, w)
        x = x.permute(0, 2, 1, 3, 4)  # (batchsize, t, c, h, w)
        x = x.reshape(x.shape[0], x.shape[2], x.shape[1], x.shape[3], x.shape[4])  # (batchsize, c, t, h, w)
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        # x = self.upsample(x)  # 使用反卷积替代线性插值
        return x[:, :self.t_prime, :, :, :]  # (batchsize, t', c, h, w)

class GraphDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, t_prime):
        super(GraphDecoder, self).__init__()
        self.t_prime = t_prime
        self.mlp1 = nn.Linear(input_dim, hidden_dim)
        self.mlp2 = nn.Linear(hidden_dim, output_dim)
        self.time_projection = nn.Linear(input_dim, t_prime)  # 修改时间投影逻辑

    def forward(self, x):
        # x: (batchsize, t, n, d)
        batchsize, t, n, d = x.shape
        x = x.reshape(batchsize * t, n, d)  # (batchsize * t, n, d)
        x = F.relu(self.mlp1(x))
        x = self.mlp2(x)  # (batchsize * t, n, c)
        x = x.reshape(batchsize, t, n, -1)  # (batchsize, t, n, c)
        x = self.time_projection(x)  # 直接通过线性层扩展时间维度
        return x[:, :self.t_prime, :, :]  # (batchsize, t', n, c)

class TimeSeriesPredictor(nn.Module):
    def __init__(self, grid_input_channels, grid_hidden_channels, grid_output_channels,
                 graph_input_dim, graph_hidden_dim, graph_output_dim, grid_t, graph_t):
        super(TimeSeriesPredictor, self).__init__()
        self.grid_encoder = GridEncoder(grid_input_channels, grid_hidden_channels, grid_output_channels)
        self.graph_encoder = GraphEncoder(graph_input_dim, graph_hidden_dim, graph_output_dim)
        self.grid_decoder = GridDecoder(grid_output_channels, grid_hidden_channels, grid_output_channels, grid_t)
        self.graph_decoder = GraphDecoder(graph_output_dim, graph_hidden_dim, graph_output_dim, graph_t)

    def forward(self, grid_input, graph_input):
        # Encode
        grid_encoded = self.grid_encoder(grid_input)  # (batchsize, output_channels, t, h, w)
        graph_encoded = self.graph_encoder(graph_input)  # (batchsize, t, n, output_dim)

        # Decode
        grid_pred = self.grid_decoder(grid_encoded)  # (batchsize, t', c, h, w)
        graph_pred = self.graph_decoder(graph_encoded)  # (batchsize, t', n, c)

        return grid_pred, graph_pred
if __name__ == "__main__":
    # Example usage
    batchsize = 32
    t = 10
    t_prime = 5
    d = 3
    h = 128
    w = 256
    n = 5612
    c = 1

    grid_input = torch.randn(batchsize, t, d, h, w)
    graph_input = torch.randn(batchsize, t, n, d)

    model = TimeSeriesPredictor(
        grid_input_channels=d,
        grid_hidden_channels=64,
        grid_output_channels=c,
        graph_input_dim=d,
        graph_hidden_dim=64,
        graph_output_dim=c,
        grid_t=t_prime,
        graph_t=t_prime
    )

    grid_pred, graph_pred = model(grid_input, graph_input)
    print(grid_pred.shape)  # (batchsize, t', c, h, w)
    print(graph_pred.shape)  # (batchsize, t', n, c)