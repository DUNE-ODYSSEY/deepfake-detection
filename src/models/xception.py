"""XceptionNet baseline — the standard FF++ detector (Rossler et al., 2019)."""
import timm
import torch.nn as nn


class XceptionDetector(nn.Module):
    IMG_SIZE = 299

    def __init__(self, pretrained: bool = True, dropout: float = 0.2):
        super().__init__()
        self.backbone = timm.create_model("legacy_xception", pretrained=pretrained,
                                          num_classes=0)  # 2048-d pooled features
        self.head = nn.Sequential(nn.Dropout(dropout),
                                  nn.Linear(self.backbone.num_features, 1))

    def forward(self, x):
        return self.head(self.backbone(x)).squeeze(-1)  # logits
