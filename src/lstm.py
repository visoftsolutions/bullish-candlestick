import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        if x.dim() == 2:  # Check if the input tensor is 2-D
            h0 = torch.zeros(self.num_layers, self.hidden_dim)
            c0 = torch.zeros(self.num_layers, self.hidden_dim)
        else:  # Input tensor is 3-D
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim)

        out, _ = self.lstm(x, (h0, c0))

        # Check if out is 2D or 3D and handle it accordingly
        if out.dim() == 3:
            out = out[:, -1, :]
        else:
            out = out[-1, :]

        out = self.linear(out)
        return out
