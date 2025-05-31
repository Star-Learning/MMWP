import torch
import torch.nn as nn
import torch.nn.functional as F

class GridEncoder(nn.Module):
    def __init__(self, input_channels, hidden_channels):
        super(GridEncoder, self).__init__()
        self.conv1 = nn.Conv3d(input_channels, hidden_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv2 = nn.Conv3d(hidden_channels, hidden_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1))

    def forward(self, x):
        # x: (batchsize, t, d, h, w)
        x = x.permute(0, 2, 1, 3, 4)  # (batchsize, d, t, h, w)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.permute(0, 2, 1, 3, 4) # (batchsize, t, output_channels, h, w)
        return x  # (batchsize, t, output_channels, h, w)

class GraphEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GraphEncoder, self).__init__()
        self.mlp1 = nn.Linear(input_dim, hidden_dim)
        self.mlp2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        # x: (batchsize, t, n, d)
        batchsize, t, n, d = x.shape
        x = x.reshape(batchsize * t, n, d)  # (batchsize * t, n, d)
        x = F.relu(self.mlp1(x))
        x = F.relu(self.mlp2(x))
        x = x.reshape(batchsize, t, n, -1)  # (batchsize, t, n, output_dim)
        return x

class GridDecoder(nn.Module):
    def __init__(self, hidden_channels, output_channels, input_t, output_t):
        super(GridDecoder, self).__init__()
        self.conv1 = nn.Conv3d(hidden_channels, hidden_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv2 = nn.Conv3d(hidden_channels, output_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        # self.upsample = nn.ConvTranspose3d(output_channels, output_channels, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=1)
        self.head_t = nn.Linear(input_t, output_t)

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4) # (batchsize, output_channels, t , h, w)
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        # x = self.upsample(x)  # 使用反卷积替代线性插值

        x = x.permute(0, 1, 3, 4, 2)
        x = self.head_t(x)
        x = x.permute(0, 4, 1, 2, 3)

        return x # (batchsize, t', c, h, w)

class GraphDecoder(nn.Module):
    def __init__(self, hidden_dim, output_dim, input_t, output_t):
        super(GraphDecoder, self).__init__()
        self.mlp1 = nn.Linear(hidden_dim, hidden_dim)
        self.mlp2 = nn.Linear(hidden_dim, output_dim)
        self.time_projection = nn.Linear(input_t, output_t)  # 修改时间投影逻辑

    def forward(self, x):
        # x: (batchsize, t, n, d)
        batchsize, t, n, d = x.shape
        x = x.reshape(batchsize * t, n, d)  # (batchsize * t, n, d)
        x = F.relu(self.mlp1(x))
        x = self.mlp2(x)  # (batchsize * t, n, c)
        x = x.reshape(batchsize, t, n, -1)  # (batchsize, t, n, c)

        x = x.permute(0, 2, 3, 1)  # (batchsize, n, t, c)
        x = self.time_projection(x)  # 直接通过线性层扩展时间维度
        x = x.permute(0, 3, 1, 2)
        return x

class TimeSeriesPredictor(nn.Module):
    def __init__(self, grid_input_channels, grid_hidden_channels, grid_output_channels,
                 graph_input_dim, graph_hidden_dim, graph_output_dim, grid_t_in, grid_t_out, graph_t_in, graph_t_out):
        super(TimeSeriesPredictor, self).__init__()
        self.grid_encoder = GridEncoder(grid_input_channels, grid_hidden_channels)
        self.graph_encoder = GraphEncoder(graph_input_dim, graph_hidden_dim)
        self.grid_decoder = GridDecoder(grid_hidden_channels, grid_output_channels, grid_t_in, grid_t_out)
        self.graph_decoder = GraphDecoder(graph_hidden_dim, graph_output_dim, graph_t_in, graph_t_out)

    def forward(self, grid_input, graph_input):
        # Encode
        grid_encoded = self.grid_encoder(grid_input)  # (batchsize, output_channels, t, h, w)
        graph_encoded = self.graph_encoder(graph_input)  # (batchsize, t, n, output_dim)

        # print('grid encoder:',grid_encoded.shape)
        # print('graph encoder:',graph_encoded.shape)
        # Decode
        grid_pred = self.grid_decoder(grid_encoded)  # (batchsize, t', c, h, w)
        graph_pred = self.graph_decoder(graph_encoded)  # (batchsize, t', n, c)

        return grid_pred, graph_pred
    
if __name__ == "__main__":
    # Example usage
    batchsize = 32
    t = 6
    t_prime = 6
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
        grid_t_in = t,
        grid_t_out = t_prime,
        graph_t_in = t,
        graph_t_out = t_prime
    )

    grid_pred, graph_pred = model(grid_input, graph_input)
    print(grid_pred.shape)  # (batchsize, t', c, h, w)
    print(graph_pred.shape)  # (batchsize, t', n, c)