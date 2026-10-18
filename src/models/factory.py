from .cnn_vit import CNNViT
from .timm_backbones import EfficientNetB4Detector, ResNet50Detector
from .xception import XceptionDetector

MODELS = {"xception": XceptionDetector, "cnn_vit": CNNViT,
          "resnet50": ResNet50Detector, "efficientnet_b4": EfficientNetB4Detector}


def create_model(name: str, pretrained: bool = True):
    if name not in MODELS:
        raise ValueError(f"Unknown model '{name}'. Choices: {list(MODELS)}")
    return MODELS[name](pretrained=pretrained)


def model_img_size(name: str) -> int:
    return MODELS[name].IMG_SIZE
