import torch

from src.longformer import SlidingWindowMHA


def test_sliding_window_mha():
    batch_size = 4
    d_model = 32
    n_head = 2
    block_size = 4

    tokens = torch.randn(batch_size, block_size, d_model)

    mha = SlidingWindowMHA(d_model, n_head)
    out = mha(tokens, tokens, tokens)

    assert out.size() == (batch_size, block_size, d_model)
