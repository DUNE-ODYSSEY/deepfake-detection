"""ImageNet-pretrained timm backbones with a binary head, at Xception's 299 px so
input resolution is not a confound in the architecture comparison (Phase 5)."""
import timm
import torch.nn as nn


class TimmDetector(nn.Module):
    IMG_SIZE = 299
    TIMM_NAME = None

    def __init__(self, pretrained: bool = True, dropout: float = 0.2):
        super().__init__()
        self.backbone = timm.create_model(self.TIMM_NAME, pretrained=pretrained, num_classes=0)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.backbone.num_features, 1))

    def forward(self, x):
        return self.head(self.backbone(x)).squeeze(-1)


class ResNet50Detector(TimmDetector):
    TIMM_NAME = "resnet50.a1_in1k"


class EfficientNetB4Detector(TimmDetector):
    TIMM_NAME = "efficientnet_b4.ra2_in1k"
