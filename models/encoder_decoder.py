import torch
import torch.nn as nn


class Cell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super(Cell, self).__init__()
        padding = kernel_size // 2
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.conv = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias
        )

    def forward(self, x, h_prev, c_prev):
        combined = torch.cat([x, h_prev], dim=1) 
        conv_output = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(conv_output, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c = f * c_prev + i * g
        h = o * torch.tanh(c)
        return h, c



class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, input_seq_len):
        super(Encoder, self).__init__()
        self.input_seq_len = input_seq_len
        self.cell = Cell(input_dim, hidden_dim, kernel_size)

    def forward(self, input_seq):
        batch_size, _, _, height, width = input_seq.size()
        h, c = (torch.zeros(batch_size, self.cell.hidden_dim, height, width, device=input_seq.device),
                torch.zeros(batch_size, self.cell.hidden_dim, height, width, device=input_seq.device))
        for t in range(self.input_seq_len):
            h, c = self.cell(input_seq[:, t], h, c)
        return h, c



class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, output_dim, output_seq_len):
        super(Decoder, self).__init__()
        self.output_seq_len = output_seq_len
        self.cell = Cell(input_dim, hidden_dim, kernel_size)
        self.output_conv = nn.Conv2d(hidden_dim, output_dim, kernel_size=1)

    def forward(self, h, c):
        batch_size, _, height, width = h.size()
        outputs = []
        x = torch.zeros(batch_size, self.cell.input_dim, height, width, device=h.device)
        for _ in range(self.output_seq_len):
            h, c = self.cell(x, h, c)
            y = self.output_conv(h)  # 
            outputs.append(y)

        return torch.stack(outputs, dim=1)

class EncoderDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, input_seq_len, output_seq_len,  kernel_size=3):
        super(EncoderDecoder, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, kernel_size, input_seq_len)
        self.decoder = Decoder(input_dim, hidden_dim, kernel_size, output_dim, output_seq_len)

    def forward(self, input_seq):
        h, c = self.encoder(input_seq)
        output_seq += self.decoder(h, c)
        return output_seq
 
