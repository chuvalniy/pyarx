import torch

from src.attention import MultiQueryAttention


def test_multiquery_attention():
    batch_size = 4
    d_model = 16
    n_head = 2
    block_size = 4

    tokens = torch.randn(batch_size, block_size, d_model)

    mha = MultiQueryAttention(d_model, n_head)
    out = mha(tokens, tokens, tokens)

    assert out.size() == (batch_size, block_size, d_model)
