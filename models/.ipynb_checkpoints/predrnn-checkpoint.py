import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import xarray as xr

# PredRNN 中的核心单元 -- SpatioTemporal LSTM Cell
class SpatioTemporalLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super(SpatioTemporalLSTMCell, self).__init__()
        self.hidden_dim = hidden_dim
        padding = kernel_size // 2
        self.conv = nn.Conv2d(input_dim + hidden_dim, hidden_dim * 7, kernel_size, padding=padding)

    def forward(self, x, h, c, m):
        combined = torch.cat([x, h, m], dim=1)
        conv_out = self.conv(combined)
        i, f, o, g, mi, mf, mo = torch.split(conv_out, self.hidden_dim, dim=1)

        i, f, o, g = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o), torch.tanh(g)
        mi, mf, mo = torch.sigmoid(mi), torch.sigmoid(mf), torch.sigmoid(mo)

        next_c = f * c + i * g
        next_m = mf * m + mi * torch.tanh(next_c)
        next_h = o * torch.tanh(next_c) + mo * torch.tanh(next_m)

        return next_h, next_c, next_m

# PredRNN 模型
class PredRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, seq_len, pred_len):
        super(PredRNN, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.hidden_dim = hidden_dim

        self.stlstm = SpatioTemporalLSTMCell(input_dim, hidden_dim)
        self.decoder = nn.Conv2d(hidden_dim, input_dim, kernel_size=1)

    def forward(self, x):
        batch_size, seq_len, channels, height, width = x.size()
        h = torch.zeros(batch_size, self.hidden_dim, height, width, device=x.device)
        c = torch.zeros_like(h)
        m = torch.zeros_like(h)

        outputs = []
        for t in range(seq_len + self.pred_len):
            if t < seq_len:
                input_frame = x[:, t]
            else:
                input_frame = output

            h, c, m = self.stlstm(input_frame, h, c, m)
            output = self.decoder(h)
            outputs.append(output)

        return torch.stack(outputs[-self.pred_len:], dim=1)