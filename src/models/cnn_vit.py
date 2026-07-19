"""Hybrid CNN-ViT detector.

Design: a pretrained CNN (ResNet-50 stages 1-3, stride 16) extracts local texture
features; its 14x14 feature map is projected to tokens and passed through a
Transformer encoder whose global self-attention models long-range spatial
inconsistencies (lighting mismatch, boundary blending, geometric warps) that
purely convolutional detectors miss. A CLS token feeds the binary head.
"""
import timm
import torch
import torch.nn as nn


class CNNViT(nn.Module):
    IMG_SIZE = 224

    def __init__(self, pretrained: bool = True, embed_dim: int = 512,
                 depth: int = 6, num_heads: int = 8, mlp_ratio: float = 4.0,
                 dropout: float = 0.1):
        super().__init__()
        # out_indices=(3,) -> C4 feature map: (B, 1024, 14, 14) for 224x224 input
        self.cnn = timm.create_model("resnet50", pretrained=pretrained,
                                     features_only=True, out_indices=(3,))
        cnn_dim = self.cnn.feature_info.channels()[-1]  # 1024
        self.proj = nn.Conv2d(cnn_dim, embed_dim, kernel_size=1)

        num_patches = (self.IMG_SIZE // 16) ** 2  # 196
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, 1)

    def forward(self, x):
        feat = self.cnn(x)[-1]                      # (B, 1024, 14, 14)
        tokens = self.proj(feat).flatten(2).transpose(1, 2)  # (B, 196, D)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        tokens = torch.cat([cls, tokens], dim=1) + self.pos_embed
        tokens = self.encoder(tokens)
        return self.head(self.norm(tokens[:, 0])).squeeze(-1)  # logits
