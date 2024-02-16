import math

import torch
import torch.nn as nn


class FeedForward(nn.Module):
    pass


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head

        self.query = nn.Linear(d_model, d_model, bias=False)
        self.key = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)

        self.projection = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v):
        B, T, C = q.shape

        q_proj = self.query(q)
        q_proj = q_proj.reshape((B, T, self.n_head, self.head_dim)).transpose(1, 2)

        k_proj = self.key(k)
        k_proj = k_proj.reshape((B, T, self.n_head, self.head_dim)).transpose(1, 2)

        v_proj = self.value(v)
        v_proj = v_proj.reshape((B, T, self.n_head, self.head_dim)).transpose(1, 2)

        wei = torch.matmul(q_proj, torch.transpose(k_proj, 2, 3)) / math.sqrt(self.head_dim)
        wei = torch.matmul(
            self.dropout(torch.nn.functional.softmax(wei, dim=-1)),
            v_proj
        )

        wei = wei.transpose(1, 2).contiguous().reshape(B, T, C)
        wei = self.projection(wei)

        return wei


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


class TransformerEncoder(nn.Module):
    def __init__(self):
        super(TransformerEncoder, self).__init__()
