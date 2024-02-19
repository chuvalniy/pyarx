import math

import torch
import torch.nn as nn
from longformer import SlidingWindowMHA

class FeedForward(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(FeedForward, self).__init__()

        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.linear2 = nn.Linear(d_model * 4, d_model)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.relu(self.linear1(x))
        out = self.linear2(out)

        return self.dropout(out)


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
            torch.nn.functional.softmax(wei, dim=-1),
            v_proj
        )

        wei = wei.transpose(1, 2).contiguous().reshape(B, T, C)
        wei = self.projection(wei)

        return self.dropout(wei)


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


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1):
        super(TransformerBlock, self).__init__()

        self.mha = MultiHeadAttention(d_model, n_head, dropout)
        self.mha_ln = nn.LayerNorm(d_model)

        self.ff = FeedForward(d_model, dropout)
        self.ff_ln = nn.LayerNorm(d_model)

    def forward(self, x):
        mha_out = self.mha(x, x, x)
        mha_out = self.mha_ln(x + mha_out)

        ff_out = self.ff(mha_out)
        ff_out = self.ff_ln(mha_out + ff_out)

        return ff_out


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, vocab_size, n_layer, n_head, dropout=0.1):
        super(TransformerEncoder, self).__init__()

        self.emb = nn.Embedding(vocab_size, d_model)
        self.pe = PositionalEncoding(d_model, dropout=dropout)
        self.pe_dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_head, dropout) for _ in range(n_layer)
        ])

    def forward(self, x):
        output = self.emb(x)
        output = self.pe_dropout(output + self.pe(output))

        for layer in self.layers:
            output = layer(output)

        return output
