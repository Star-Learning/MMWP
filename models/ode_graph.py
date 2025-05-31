from torch import nn
from torchdiffeq import odeint_adjoint as odeint
import torch

class ODEFunc(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        # 极简残差结构 (参数量不变但更稳定)
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.norm = nn.LayerNorm(hidden_dim)  # 更适合序列数据的归一化

    def forward(self, t, x):
        return x + self.norm(self.net(x))  # 修正后的标准残差结构

class ODEBlock(nn.Module):
    def __init__(self, odefunc, method='dopri5', rtol=1e-3, atol=1e-4):  # 改用自适应求解器
        super().__init__()
        self.odefunc = odefunc
        self.method = method
        self.rtol = rtol
        self.atol = atol

    def forward(self, x0, t):
        out = odeint(self.odefunc, x0, t, 
                    method=self.method,
                    rtol=self.rtol,
                    atol=self.atol)
        return out.permute(1, 0, 2, 3)  # (B, T_out, N, C)

class ODEForecastModelGraph(nn.Module):
    def __init__(self, feature_dim, T_out):
        super().__init__()
        # 保持原有参数定义不变
        self.encoder = nn.Linear(feature_dim, feature_dim)
        self.odeblock = ODEBlock(ODEFunc(feature_dim))
        self.decoder = nn.Linear(feature_dim, feature_dim)
        self.T_out = T_out

    def forward(self, x):
        B, T_in, N, C = x.shape
        x = self.encoder(x)
        x0 = x[:, -1]  # 保持原有时间步选择逻辑
        t = torch.linspace(0, T_in/self.T_out, self.T_out, device=x.device)  # 动态时间缩放
        pred = self.odeblock(x0, t)
        return self.decoder(pred)

# 测试代码保持完全一致
if __name__ == '__main__':
    batch_size = 2
    T_in = 96
    T_out = 720
    N = 5000
    C = 1
    model = ODEForecastModelGraph(feature_dim=C, T_out=T_out)

    x = torch.randn(batch_size, T_in, N, C)
    y = model(x)
    print(y.shape)  # 输出: torch.Size([2, 6, 5000, 1])