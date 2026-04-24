"""Небольшой беcякорный детектор в стиле YOLO.

Backbone: ResNet-18, урезанный до layer3 (stride 16, 256 каналов),
          инициализированный весами ImageNet из torchvision.
Neck:     свёртка 3x3 + BN + SiLU, 256 каналов.
Head:     свёртка 1x1, выдающая 5 + num_classes каналов на каждую ячейку сетки:
            - 1 логит objectness (наличие объекта)
            - 4 параметра бокса (cx, cy, w, h), каждый в [0, 1] через sigmoid,
              интерпретируются относительно всего (letterboxed) изображения
            - num_classes логитов классов

Выход:
    {
      "obj":  [B, H, W]            -- сырые логиты (до sigmoid)
      "bbox": [B, H, W, 4]         -- cxcywh после sigmoid, в [0, 1]
      "cls":  [B, H, W, C]         -- сырые логиты классов
    }
"""
import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights


class YOLO(nn.Module):
    def __init__(self, num_classes: int = 1, pretrained: bool = True, stride: int = 16):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        net = resnet18(weights=weights)
        self.backbone = nn.Sequential(
            net.conv1, net.bn1, net.relu, net.maxpool,
            net.layer1, net.layer2, net.layer3,       # stride 16, 256 каналов
        )
        self.neck = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.SiLU(inplace=True),
        )
        # 5 = 1 obj + 4 bbox
        self.head = nn.Conv2d(256, 5 + num_classes, 1)
        self.num_classes = num_classes
        self.stride = stride

        # Смещаем логиты objectness в отрицательную сторону при инициализации,
        # чтобы модель изначально предсказывала "нет объекта" почти везде (sigmoid(-4) ~ 0.018).
        nn.init.constant_(self.head.bias[0], -4.0)

    def forward(self, x: torch.Tensor):
        f = self.backbone(x)                                  # [B, 256, H, W]
        f = self.neck(f)
        out = self.head(f).permute(0, 2, 3, 1).contiguous()   # [B, H, W, 5+C]
        return {
            "obj":  out[..., 0],                              # сырые логиты
            "bbox": out[..., 1:5].sigmoid(),                  # координаты в [0, 1]
            "cls":  out[..., 5:],                             # логиты классов
        }
