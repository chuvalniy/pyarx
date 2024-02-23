import math

import torch
import torch.nn as nn


class MultiQueryAttention(nn.Module):
    """
    Attention where splitting into heads applied only for queries.

    arXiv: https://arxiv.org/pdf/1911.02150.pdf
    """

    def __init__(self, d_model, n_head, dropout=0.1):
        super(MultiQueryAttention, self).__init__()

        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = self.d_model // self.n_head

        self.dropout = nn.Dropout(dropout)

        self.query = nn.Linear(d_model, d_model, bias=False)
        self.key = nn.Linear(d_model, self.head_dim, bias=False)
        self.value = nn.Linear(d_model, self.head_dim, bias=False)

        self.projection = nn.Linear(d_model, d_model)

    def forward(self, q, k, v):
        B, T, C = q.shape

        q_proj = self.query(q)
        q_proj = q_proj.reshape((B, T, self.n_head, self.head_dim)).transpose(1, 2)

        k_proj = self.key(k)
        k_proj = k_proj.reshape((B, T, 1, self.head_dim)).transpose(1, 2)

        v_proj = self.value(v)
        v_proj = v_proj.reshape((B, T, 1, self.head_dim)).transpose(1, 2)

        wei = torch.matmul(q_proj, torch.transpose(k_proj, 2, 3)) / math.sqrt(self.head_dim)

        wei = torch.matmul(
            torch.nn.functional.softmax(wei, dim=-1),
            v_proj
        )

        wei = wei.transpose(1, 2).contiguous().reshape(B, T, C)
        wei = self.projection(wei)

        return self.dropout(wei)
