
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
            "obj":  out[..., 0],        # сырые логиты
            "bbox": out[..., 1:5],      # сырые логиты боксов (декодируются через decode_boxes)
            "cls":  out[..., 5:],       # логиты классов
        }
