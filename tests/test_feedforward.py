import torch
from src.transformer import FeedForward


def test_feedforward():
    batch_size = 4
    d_model = 4
    block_size = 4

    logits = torch.randn(batch_size, block_size, d_model)
    ff = FeedForward(d_model)

    out = ff(logits)
    assert out.size() == (batch_size, block_size, d_model)