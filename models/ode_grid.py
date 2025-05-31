import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint

class ODEFunc(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1), 
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1)
        )
        self.norm = nn.GroupNorm(min(4, channels), channels)  

    def forward(self, t, x):
        return x + self.norm(self.conv(x))


class ODEForecastModelGrid(nn.Module):
    def __init__(self, channels, T_out, method='dopri5', rtol=1e-3, atol=1e-4):
        super().__init__()
        self.channels = channels
        self.T_out = T_out
        self.ode_func = ODEFunc(channels)
        self.method = method
        self.rtol = rtol
        self.atol = atol

    def forward(self, x, t=None):
        bs, T_in, c, h, w = x.shape
        x = x.permute(1, 0, 2, 3, 4)
        x0 = x[0]

        time_scale = T_in / self.T_out * 5
        if t is None:
            t = torch.linspace(0, time_scale, self.T_out, device=x.device)

        out = odeint(self.ode_func, x0, t, 
                   method=self.method,
                   rtol=self.rtol,
                   atol=self.atol)
        
        return out.permute(1, 0, 2, 3, 4)
