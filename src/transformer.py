import math

import torch
import torch.nn as nn

from src.attention import MultiHeadAttention


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
