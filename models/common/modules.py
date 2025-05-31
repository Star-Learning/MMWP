import torch
import torch.nn as nn

class DecoderGrid(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.1):
        super(DecoderGrid, self).__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(in_channels),
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels)
        )
        self.norm_out = nn.LayerNorm(out_channels)

    def forward(self, x):
        b, t, c, h, w = x.shape

        x_reshaped = x.permute(0, 1, 3, 4, 2).contiguous().view(-1, c)

        x_mlp = self.mlp(x_reshaped)

        x_mlp = self.norm_out(x_mlp)

        out_c = x_mlp.shape[-1]
        x_out = x_mlp.view(b, t, h, w, out_c).permute(0, 1, 4, 2, 3)

        return x_out

class DecoderGraph(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.1):
        super(DecoderGraph, self).__init__()
        self.norm_in = nn.LayerNorm(in_channels)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels)
        )
        self.norm_out = nn.LayerNorm(out_channels)

    def forward(self, x):
        # x shape: (b, t, n, c)
        b, t, n, c = x.shape

        # flatten to (b * t * n, c)
        x_reshaped = x.reshape(-1, c)
        x_norm = self.norm_in(x_reshaped)
        x_mlp = self.mlp(x_norm)
        x_out = self.norm_out(x_mlp)

        # reshape back to (b, t, n, out_c)
        out_c = x_out.shape[-1]
        return x_out.view(b, t, n, out_c)



class MLPTimeGraph(nn.Module):
    def __init__(self, t_in, t_out):
        super(MLPTimeGraph, self).__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(t_in),
            nn.Linear(t_in, t_out), 
            nn.ReLU()
        )

    def forward(self, x):
        # x: (b, t, n, c)
        b, t, n, c = x.shape

        x = x.permute(0, 2, 3, 1)
        x = x.reshape(-1, t)
        x = self.proj(x)  # (b*n*c, t')

        t_out = x.shape[-1]
        x = x.view(b, n, c, t_out)

        x = x.permute(0, 3, 1, 2)

        return x
    

class MLPTimeGrid(nn.Module):
    def __init__(self, t_in, t_out):
        super(MLPTimeGrid, self).__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(t_in),
            nn.Linear(t_in, t_out),
            nn.ReLU()
        )

    def forward(self, x):
        # x: (b, t, c, h, w)
        b, t, c, h, w = x.shape

        x = x.permute(0, 2, 3, 4, 1)
        x = x.reshape(-1, t)
        x = self.proj(x)  # (b*c*h*w, t')

        t_out = x.shape[-1]
        x = x.view(b, c, h, w, t_out)

        x = x.permute(0, 4, 1, 2, 3)

        return x


class SpatioTemporalDecoderGrid(nn.Module):
    def __init__(self, in_channels, hidden_dim, num_layers=4, num_heads=8):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        x = self.conv1(x)  # -> (B*T, hidden_dim, H, W)
        x = x.flatten(2).transpose(1, 2)  
        x = self.transformer(x)  
        x = x.transpose(1, 2).view(B, T, -1, H, W) 
        return x

class SpatioTemporalDecoderGraph(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers=4, num_heads=8):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        # x: (B, T, N, C)
        B, T, N, C = x.shape
        x = self.input_proj(x)  # (B, T, N, hidden_dim)
        x = x.permute(0, 2, 1, 3).contiguous()  # (B, N, T, hidden_dim)
        x = x.view(B * N, T, -1)  # flatten N
        x = self.transformer(x)  # (B*N, T, hidden_dim)
        x = x.view(B, N, T, -1).permute(0, 2, 1, 3)  # (B, T, N, hidden_dim)
        return x
