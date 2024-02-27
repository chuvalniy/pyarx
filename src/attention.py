import math

import torch
import torch.nn as nn


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


class GroupedQueryAttention(nn.Module):
    """
    arXiv: https://arxiv.org/abs/2305.13245
    """

    def __init__(self, d_model, n_head, n_group, dropout=0.1):
        super(GroupedQueryAttention, self).__init__()

        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head

        self.n_group = n_group
        self.head_in_group = n_head // n_group

        self.query = nn.Linear(d_model, d_model, bias=False)
        self.key = nn.Linear(d_model, self.head_dim * self.n_group, bias=False)
        self.value = nn.Linear(d_model, self.head_dim * self.n_group, bias=False)

        self.projection = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v):
        B, T, C = q.shape

        q_proj = self.query(q)

        q_proj = q_proj.reshape((B, T, self.n_head, self.head_dim)).transpose(1, 2)
        q_proj = q_proj.reshape((B, self.head_in_group, self.n_group, T, self.head_dim))

        k_proj = self.key(k)
        k_proj = k_proj.reshape((B, T, self.n_group, 1, self.head_dim)).transpose(1, 3)

        v_proj = self.value(v)
        v_proj = v_proj.reshape((B, T, self.n_group, 1, self.head_dim)).transpose(1, 3)

        wei = torch.matmul(q_proj, torch.transpose(k_proj, -2, -1)) / math.sqrt(self.head_dim)

        wei = torch.matmul(
            torch.nn.functional.softmax(wei, dim=-1),
            v_proj
        )

        wei = wei.reshape((B, self.n_head, T, self.head_dim))
        wei = wei.transpose(1, 2).contiguous().reshape(B, T, C)
        wei = self.projection(wei)

        return self.dropout(wei)



class SlidingWindowMHA(nn.Module):
    """
    Sliding window approach for self-attention mechanism.
    Reference: https://arxiv.org/pdf/2004.05150.pdf
    """

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

        wei = torch.matmul(
            torch.nn.functional.softmax(wei, dim=-1),
            v_proj
        )

        wei = wei.transpose(1, 2).contiguous().reshape(B, T, C)
        wei = self.projection(wei)

        return self.dropout(wei)
