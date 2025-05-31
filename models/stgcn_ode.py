import torch
import torch.nn as nn
import torch.nn.functional as F

from torchdiffeq import odeint


class ODEFunc(nn.Module):
    def __init__(self, channels, num_nodes):
        super(ODEFunc, self).__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=(1, 1))
        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()

    def forward(self, t, x):
        dx = self.conv(x)
        dx = self.bn(dx)
        return self.relu(dx)


class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, use_ode=False, num_nodes=None):
        super(STGCNBlock, self).__init__()
        padding = (kernel_size - 1) // 2
        self.temporal_conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(padding, 0))
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(out_channels)
        self.use_ode = use_ode
        self.ode_func = ODEFunc(out_channels, num_nodes) if use_ode else None

    def graph_conv(self, x, A_hat):
        # x: (B, C, T, N), A_hat: sparse tensor shape (N, N)
        x = x.permute(0, 2, 3, 1)  # (B, T, N, C)
        B, T, N, C = x.shape
        x = x.reshape(B * T, N, C)  # (B*T, N, C)
        x = torch.bmm(A_hat.to_dense().unsqueeze(0).repeat(B * T, 1, 1), x)  # (B*T, N, C)
        x = x.reshape(B, T, N, C).permute(0, 3, 1, 2)  # (B, C, T, N)
        return x

    def forward(self, x, A_hat):
        x = self.temporal_conv(x)
        x = self.relu(x)
        x = self.graph_conv(x, A_hat)

        if self.use_ode:
            t = torch.tensor([0, 1], dtype=torch.float32, device=x.device)
            x = odeint(self.ode_func, x, t)[-1]

        x = self.bn(x)
        return self.relu(x)

# ------------------ STGCN with ODE ------------------ #
class STGCN(nn.Module):
    def __init__(self, in_channels, out_channels, num_nodes, target_nodes, t_in, t_out, hidden_channels=32, use_ode=False):
        super(STGCN, self).__init__()
        self.t_out = t_out
        self.num_nodes = num_nodes
        self.target_nodes = target_nodes
        self.block1 = STGCNBlock(in_channels, hidden_channels, use_ode=use_ode, num_nodes=num_nodes)
        self.block2 = STGCNBlock(hidden_channels, hidden_channels, use_ode=use_ode, num_nodes=num_nodes)
        self.final_temporal = nn.Conv2d(hidden_channels, out_channels * t_out, kernel_size=(1, 1))

    def forward(self, x, A_hat):
        # x: (B, T, N, C)
        x = x.permute(0, 3, 1, 2)  # -> (B, C, T, N)
        x = self.block1(x, A_hat)
        x = self.block2(x, A_hat)
        x = self.final_temporal(x)  # -> (B, C_out * T_out, T, N)
        B, CoutT, T, N = x.shape
        x = x.permute(0, 2, 3, 1)  # (B, T, N, C_out * T_out)
        x = x.reshape(B, T, N, self.t_out, -1)  # (B, T, N, T_out, C_out)
        x = x[:, -1]  # 取最后一个时间步作为输出 (B, N, T_out, C_out)
        x = x[:, :self.target_nodes]  # 取目标节点
        return x.permute(0, 2, 1, 3)  # (B, T_out, target_nodes, C_out)
    
