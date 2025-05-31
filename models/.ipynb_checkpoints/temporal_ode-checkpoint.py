import torch
import torch.nn as nn
import torch.nn.functional as F

from torchdiffeq import odeint

class CNNDynamics(nn.Module):
    def __init__(self, c):
        super(CNNDynamics, self).__init__()
        self.c = c
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, c, kernel_size=3, padding=1)

    def forward(self, t, x):
        # x shape: (batch_size, c, h, w)
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        return out

class ODEModel(nn.Module):
    def __init__(self, time_steps, pde, time_interval=1.0, method="rk4", rtol=1e-5, atol=1e-5):
        super(ODEModel, self).__init__()
        self.pde = pde 
        self.time_steps = time_steps
        self.time_interval = time_interval
        self.method = method
        self.rtol = rtol
        self.atol = atol

    def forward(self, x):
        # x shape: (batch_size, t, c, h, w)
        x0 = x[:, 0, :, :, :]
        time_points = torch.linspace(0, self.time_interval * (self.time_steps - 1), self.time_steps).to(x.device)
        ode_output = odeint(self.pde, x0, time_points, method=self.method, rtol=self.rtol, atol=self.atol)
        ode_output = ode_output.permute(1, 0, 2, 3, 4)
        return ode_output

if __name__ == '__main__':
    batch_size = 4
    t = 8  
    c = 42
    h = 32
    w = 64

    input_data = torch.randn(batch_size, t, c, h, w).cuda()
    cnn_dynamics = CNNDynamics(c=c).cuda()
    ode_model = ODEModel(time_steps=t, pde=cnn_dynamics, time_interval=1.0).cuda()
    output = ode_model(input_data)
    print("Output shape:", output.shape) # (batch_size, t, c, h, w)