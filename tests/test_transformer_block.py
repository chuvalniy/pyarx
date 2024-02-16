import torch

from src.transformer import TransformerBlock


def test_transformer_block():
    batch_size = 4
    block_size = 4
    d_model = 8
    n_head = 2

    logits = torch.randn(batch_size, block_size, d_model)
    transformer_block = TransformerBlock(d_model, n_head)

    out = transformer_block(logits)
    assert out.size() == (batch_size, block_size, d_model)
