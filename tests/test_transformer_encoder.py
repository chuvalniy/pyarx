from src.transformer import TransformerEncoder
import torch


def test_transformer_encoder():
    batch_size = 4
    block_size = 4
    d_model = 4
    vocab_size = 32
    n_layer = 2
    n_head = 2

    tokens = torch.randint(low=0, high=vocab_size, size=(batch_size, block_size))

    transformer = TransformerEncoder(d_model, vocab_size, n_layer, n_head)
    out = transformer(tokens)

    assert out.size() == (batch_size, block_size, d_model)