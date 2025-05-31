import torch
import torch.nn as nn
import torch.nn.functional as F

class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(STGCNBlock, self).__init__()
        padding = (kernel_size - 1) // 2
        self.temporal_conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(padding, 0))
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x, A):
        # x: (B, C, T, N), A: (N, N) or (B, N, N)
        x = self.temporal_conv(x) 
        x = self.relu(x)
        x = self.graph_conv(x, A) 
        x = self.bn(x)
        return self.relu(x)

    def graph_conv(self, x, A):
        B, C, T, N = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B * T, C, N)  # (B*T, C, N)

        if A.is_sparse:
            x_out = torch.stack([torch.sparse.mm(A, x_i.T).T for x_i in x], dim=0)
        else:
            A = A.unsqueeze(0).expand(B * T, -1, -1)  # (B*T, N, N)
            x_out = torch.bmm(x, A)

        x_out = x_out.view(B, T, C, N).permute(0, 2, 1, 3)  # (B, C, T, N)
        return x_out

class STGCN(nn.Module):
    def __init__(self, in_channels, out_channels, num_nodes, target_nodes, t_in, t_out, hidden_channels=32):
        super(STGCN, self).__init__()
        self.t_out = t_out
        self.block1 = STGCNBlock(in_channels, hidden_channels)
        self.block2 = STGCNBlock(hidden_channels, hidden_channels)
        self.final_temporal = nn.Conv2d(hidden_channels, out_channels * t_out, kernel_size=(1, 1))
        self.num_nodes = num_nodes
        self.target_nodes = target_nodes
        
    def forward(self, x, A):
        x = x.permute(0, 3, 1, 2)  # (B, C, T, N)
        x = self.block1(x, A)
        x = self.block2(x, A)
        x = self.final_temporal(x)  # (B, C_out * T_out, T, N)
        B, CtT, T, N = x.shape
        C_out = CtT // self.t_out
        x = x.view(B, self.t_out, C_out, T, N)  # (B, T_out, C_out, T, N)
        x = x.mean(3)  # 平均掉原来的 T 维 => (B, T_out, C_out, N)
        x = x.permute(0, 1, 3, 2)  # (B, T_out, N, C_out)
        x = x[:, :, :self.target_nodes, :]
        return x
    