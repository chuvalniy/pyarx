import torch
from src.transformer import MultiHeadAttention

def test_multihead_attention():
    batch_size = 4
    d_model = 16
    n_head = 2
    block_size = 4

    tokens = torch.randn(batch_size, block_size, d_model)

    mha = MultiHeadAttention(d_model, n_head)
    out = mha(tokens, tokens, tokens)

    assert out.size() == (batch_size, block_size, d_model)