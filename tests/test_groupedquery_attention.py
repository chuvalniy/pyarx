from src.attention import GroupedQueryAttention
import torch

def test_grouped_query_attention():
    batch_size = 4
    d_model = 32
    n_head = 4
    n_group = 2
    block_size = 4

    tokens = torch.randn(batch_size, block_size, d_model)

    mha = GroupedQueryAttention(d_model, n_head, n_group)
    out = mha(tokens, tokens, tokens)

    assert out.size() == (batch_size, block_size, d_model)
