"""Функция потерь в стиле YOLO: objectness (BCE) + бокс (CIoU) + класс (BCE).

Назначение целей:
  Для каждого GT-бокса на изображении находим ячейку сетки, центр которой содержит
  центр бокса — эта ячейка отвечает за предсказание. Все остальные ячейки
  получают цель objectness = 0.
"""
import torch
import torch.nn.functional as F
from torch import nn

from .utils import box_cxcywh_to_xyxy, ciou, decode_boxes


class YOLOLoss(nn.Module):
    def __init__(self, num_classes: int, w_obj: float = 1.0, w_noobj: float = 0.5,
                 w_bbox: float = 5.0, w_cls: float = 1.0):
        super().__init__()
        self.num_classes = num_classes
        self.w_obj, self.w_noobj = w_obj, w_noobj
        self.w_bbox, self.w_cls  = w_bbox, w_cls

    # ---------- назначение целей ----------
    def build_targets(self, outputs, targets):
        B, H, W = outputs["obj"].shape
        device  = outputs["obj"].device

        obj_t = torch.zeros(B, H, W, device=device)
        box_t = torch.zeros(B, H, W, 4, device=device)
        cls_t = torch.zeros(B, H, W, self.num_classes, device=device)
        mask  = torch.zeros(B, H, W, dtype=torch.bool, device=device)

        for b in range(B):
            boxes  = targets[b]["boxes"]      # [N, 4] cxcywh нормализованные
            labels = targets[b]["labels"]     # [N]
            if boxes.numel() == 0:
                continue
            gx = (boxes[:, 0] * W).clamp(0, W - 1).long()
            gy = (boxes[:, 1] * H).clamp(0, H - 1).long()
            # Более поздние боксы перезаписывают предыдущие в той же ячейке
            obj_t[b, gy, gx]         = 1.0
            box_t[b, gy, gx]         = boxes      # сохраняем оригинальные нормализованные боксы
            mask [b, gy, gx]         = True
            cls_t[b, gy, gx, labels] = 1.0
        return obj_t, box_t, cls_t, mask

    # ---------- прямой проход ----------
    def forward(self, outputs, targets):
        obj_t, box_t, cls_t, mask = self.build_targets(outputs, targets)
        neg = ~mask

        # ---- objectness (BCE) ----
        obj_logits = outputs["obj"]
        if mask.any():
            obj_pos = F.binary_cross_entropy_with_logits(
                obj_logits[mask], obj_t[mask], reduction="mean"
            )
        else:
            obj_pos = obj_logits.sum() * 0
        if neg.any():
            obj_neg = F.binary_cross_entropy_with_logits(
                obj_logits[neg], obj_t[neg], reduction="mean"
            )
        else:
            obj_neg = obj_logits.sum() * 0
        obj_loss = self.w_obj * obj_pos + self.w_noobj * obj_neg

        # ---- бокс (CIoU на совпавших ячейках) ----
        if mask.any():
            # Декодируем предсказания из смещений относительно ячеек
            pred_boxes_full = decode_boxes(outputs["bbox"])   # [B, H, W, 4] нормализованные
            pred_boxes = pred_boxes_full[mask]                # [N, 4]
            tgt_boxes  = box_t[mask]                          # [N, 4] оригинальные нормализованные
            box_loss = (1 - ciou(
                box_cxcywh_to_xyxy(pred_boxes),
                box_cxcywh_to_xyxy(tgt_boxes),
            )).mean()
        else:
            box_loss = outputs["bbox"].sum() * 0

        # ---- класс (BCE на совпавших ячейках) ----
        if mask.any():
            cls_loss = F.binary_cross_entropy_with_logits(
                outputs["cls"][mask], cls_t[mask], reduction="mean"
            )
        else:
            cls_loss = outputs["cls"].sum() * 0

        total = obj_loss + self.w_bbox * box_loss + self.w_cls * cls_loss
        return {
            "total": total,
            "obj":   obj_loss.detach(),
            "bbox":  box_loss.detach(),
            "cls":   cls_loss.detach(),
        }
