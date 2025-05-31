import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint

class ODEFunc(nn.Module):
    def __init__(self, hidden_dim):
        super(ODEFunc, self).__init__()
        self.conv1 = nn.Conv3d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.norm1 = nn.BatchNorm3d(hidden_dim)  # 添加归一化
        self.conv2 = nn.Conv3d(hidden_dim, hidden_dim, kernel_size=3, padding=1)

    def forward(self, t, x):
        out = self.conv1(x)
        out = self.norm1(out)  # 归一化
        out = self.relu(out)
        out = self.conv2(out)
        return out

class NeuralODEModel(nn.Module):
    def __init__(self, input_channels, hidden_dim, output_channels, T_prime):
        super(NeuralODEModel, self).__init__()
        self.downsample = nn.Conv3d(input_channels, hidden_dim, kernel_size=1)
        self.odefunc = ODEFunc(hidden_dim)
        self.upsample = nn.Conv3d(hidden_dim, output_channels, kernel_size=1)
        self.T_prime = T_prime
        self.t_points = torch.linspace(0., 1., T_prime)

    def forward(self, x):
        batch_size, T, channels, height, width = x.shape

        # Reshape and permute the input data to match the expected dimensions for Conv3D
        x = x.permute(0, 2, 1, 3, 4)  # Shape becomes (batch_size, channels, T, height, width)

        # Downsample the input data to the hidden dimension
        x = self.downsample(x)

        # Solve the ODE over the specified time points
        sol = odeint(self.odefunc, x, self.t_points, method='euler')

        # Upsample the solution back to the original channel size
        sol = self.upsample(sol[-1])  # Take the last solution point and upsample it

        # Permute the output back to the original shape (batch_size, T', channels, height, width)
        sol = sol.permute(0, 2, 1, 3, 4)

        # Repeat the solution along the time dimension to match T'
        sol = sol.repeat_interleave(self.T_prime // T, dim=1)

        return sol

# Example usage
if __name__ == "__main__":
    batch_size = 8
    T = 10
    T_prime = 20
    C = 3
    H = 64
    W = 64

    # Create a dummy input tensor with shape (batch_size, T, C, H, W)
    input_data = torch.randn(batch_size, T, C, H, W)

    # Initialize the model with T_prime
    model = NeuralODEModel(input_channels=C, hidden_dim=16, output_channels=C, T_prime=T_prime)

    # Perform the interpolation
    interpolated_data = model(input_data)

    print(interpolated_data.shape)  # Should be (batch_size, T_prime, C, H, W)