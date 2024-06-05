import torch.nn as nn
from einops.layers.torch import Rearrange
from einops import repeat
import torch


class ViT(nn.Module):
    def __init__(self, img_size, in_channels, patch_size, d_model, n_heads, n_layers, dropout, n_classes):
        super(ViT, self).__init__()

        self.patch_emb = PatchEmbedding(in_channels, patch_size, d_model)

        num_patches = (img_size // patch_size) ** 2
        self.pos_encoding = nn.Parameter(torch.randn(1, num_patches + 1, d_model))
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        self.attention = Attention(d_model, n_heads, dropout)

        self.layers = nn.ModuleList([])
        for _ in range(n_layers):
            transformer_block = nn.Sequential(
                ResidualAdd(PreNorm(d_model, Attention(d_model, n_heads, dropout))),
                ResidualAdd(PreNorm(d_model, FeedForward(d_model, dropout)))
            )
            self.layers.append(transformer_block)

        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, n_classes))

    def forward(self, img):
        x = self.patch_emb(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_encoding[:, :(n + 1)]

        for layer in self.layers:
            x = layer(x)

        return self.head(x[:, 0, :])


class FeedForward(nn.Sequential):
    def __init__(self, d_model, dropout):
        super(FeedForward, self).__init__(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super(ResidualAdd, self).__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class PreNorm(nn.Module):
    def __init__(self, d_model, fn):
        super(PreNorm, self).__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, patch_size, d_model):
        super(PatchEmbedding, self).__init__()

        self.projection = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(patch_size * patch_size * in_channels, d_model)
        )

    def forward(self, x):
        x = self.projection(x)
        return x


class Attention(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super(Attention, self).__init__()

        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout)

        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)

    def forward(self, x):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        out, _ = self.attention(q, k, v)
        return out


if __name__ == "__main__":
    in_channels = 3
    img_size = 128

    data = torch.randn(4, in_channels, img_size, img_size)

    model = ViT(img_size, in_channels, 16, 32, 2, 2, 0.1, 4)
    out = model(data)

    print(out.shape)
