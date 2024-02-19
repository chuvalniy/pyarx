import math

import torch
import torch.nn as nn

"""
Transformer model with sliding window approach for self-attention mechanism.
Reference: https://arxiv.org/pdf/2004.05150.pdf
"""


class SlidingWindowMHA(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1, kernel_size=1):
        super(SlidingWindowMHA, self).__init__()

        self.kernel_size = kernel_size
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

        # optimize
        wei = torch.zeros(size=(B, self.n_head, T, T))
        for i in range(T):
            for j in range(T):
                start = max(0, j - self.kernel_size)
                end = min(T, j + self.kernel_size)

                q_window = q_proj[:, :, i, start:end]
                k_window = k_proj[:, :, i, start:end]
                res = torch.matmul(
                    q_window[:, :, None, :], torch.transpose(k_window[:, :, None, :], 2, 3)
                ) / math.sqrt(self.head_dim)

                wei[:, :, i, j] = res[:, :, 0, 0]

        # query      key.T
        # ****    * * * * *
        # ****    * * * * *
        # ****    * * * * *
        # ****    * * * * *
        # ****

        # wei = torch.matmul(q_proj, torch.transpose(k_proj, 2, 3)) / torch.sqrt(self.head_dim)
        wei = torch.matmul(
            torch.nn.functional.softmax(wei, dim=-1),
            v_proj
        )

        wei = wei.transpose(1, 2).contiguous().reshape(B, T, C)
        wei = self.projection(wei)

        return self.dropout(wei)
