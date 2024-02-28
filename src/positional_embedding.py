import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_length=10_000, dropout=0.1):
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(1, max_length, d_model)
        for pos in range(max_length):
            for i in range(0, d_model, 2):
                pe[:, pos, i] = math.sin(pos * 10_000 ** (-i / d_model))
                pe[:, pos, i + 1] = math.cos(pos * 10_000 ** (-i / d_model))

        self.register_buffer('pe', pe)

    def forward(self, x):
        B, T, C = x.shape

        out = self.dropout(x + self.pe[:, :T, :])
        return out


class RotaryPositionalEmbeddings(nn.Module):
    def __init__(self, d_model, max_length=10_000):
        super(RotaryPositionalEmbeddings, self).__init__()

        cos_pe = torch.zeros(1, max_length, d_model)
        sin_pe = torch.zeros(1, max_length, d_model)
        for pos in range(max_length):
            for i in range(d_model // 2):
                cos_pe[:, pos, 2 * i] = math.cos(pos * 10_000 ** (-2 * (i - 1) / d_model))
                cos_pe[:, pos, 2 * i + 1] = math.cos(pos * 10_000 ** (-2 * (i - 1) / d_model))

                sin_pe[:, pos, i] = math.sin(pos * 10_000 ** (-2 * (i - 1) / d_model))
                sin_pe[:, pos, 2 * i + 1] = math.sin(pos * 10_000 ** (-2 * (i - 1) / d_model))

        self.register_buffer('cos_pe', cos_pe)
        self.register_buffer('sin_pe', cos_pe)

    def forward(self, x):
        _, T, _ = x.shape

        x_rotated = self.rotate(x)
        out = x * self.cos_pe[:, :T, :] + x_rotated * self.sin_pe[:, :T, :]
        return out

    def rotate(self, x):
        _, _, E = x.shape

        out = torch.zeros_like(x)
        for i in range(0, E, 2):
            out[:, :, i + 1] = x[:, :, i]
            out[:, :, i] = -x[:, :, i + 1]

        return out
