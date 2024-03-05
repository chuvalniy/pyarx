import torch
import torch.nn as nn

from src.normalization import RMSNorm


def test_positional_encoding():
    batch_size = 4
    d_model = 16
    block_size = 8
    vocab_size = 128

    tokens = torch.randint(low=0, high=vocab_size, size=(batch_size, block_size))

    embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
    rmsnorm = RMSNorm(d_model)

    embeds = embedding_layer(tokens)
    out = rmsnorm(embeds)

    assert out.size() == (batch_size, block_size, d_model)
