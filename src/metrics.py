"""Декодирование выходов модели и вычисление mAP@0.5."""
from collections import defaultdict
import torch

from .utils import box_cxcywh_to_xyxy, box_iou, nms


@torch.no_grad()
def decode_predictions(outputs: dict, img_size: int,
                        conf_thresh: float = 0.05, top_k: int = 100,
                        nms_iou: float = 0.5):
    """Конвертирует сырые выходы YOLO в списки детекций для каждого изображения в пикселях холста.

    Возвращает список словарей [{"boxes": xyxy[N,4], "scores":[N], "labels":[N]}].
    """
    B, H, W = outputs["obj"].shape
    obj = outputs["obj"].sigmoid()                    # [B, H, W]
    box = outputs["bbox"] * img_size                  # pixel cxcywh on canvas
    cls = outputs["cls"].softmax(-1)                  # [B, H, W, C]
    C   = cls.shape[-1]
    results = []

    for b in range(B):
        o    = obj[b].flatten()                       # [HW]
        bx   = box[b].view(-1, 4)                     # [HW, 4]
        cs, cl = cls[b].view(-1, C).max(-1)           # [HW], [HW]
        scores = o * cs                               # совместная уверенность
        keep = scores >= conf_thresh
        scores, bx, cl = scores[keep], bx[keep], cl[keep]
        if scores.numel() == 0:
            results.append({"boxes":  bx.new_zeros((0, 4)),
                             "scores": scores, "labels": cl})
            continue
        if scores.numel() > top_k:
            top = scores.argsort(descending=True)[:top_k]
            scores, bx, cl = scores[top], bx[top], cl[top]
        xyxy = box_cxcywh_to_xyxy(bx)
        idx  = nms(xyxy, scores, iou_thresh=nms_iou)
        results.append({
            "boxes":  xyxy[idx],
            "scores": scores[idx],
            "labels": cl[idx],
        })
    return results


@torch.no_grad()
def compute_map50(all_preds, all_targets, num_classes: int = 1,
                   iou_thresh: float = 0.5):
    """Классический mAP@0.5 со 101-точечной интерполяцией. Возвращает (mAP, по классам)."""
    per_gt     = defaultdict(int)
    per_tp     = defaultdict(list)
    per_fp     = defaultdict(list)
    per_scores = defaultdict(list)

    for pred, tgt in zip(all_preds, all_targets):
        for c in tgt["labels"].tolist():
            per_gt[int(c)] += 1
        if pred["boxes"].shape[0] == 0:
            continue

        order = pred["scores"].argsort(descending=True)
        p_boxes, p_scores, p_labels = pred["boxes"][order], pred["scores"][order], pred["labels"][order]

        for c in range(num_classes):
            pm = p_labels == c
            tm = tgt["labels"] == c
            pb, ps = p_boxes[pm], p_scores[pm]
            tb = tgt["boxes"][tm]
            if pb.shape[0] == 0:
                continue
            if tb.shape[0] == 0:
                per_tp[c].extend([0] * pb.shape[0])
                per_fp[c].extend([1] * pb.shape[0])
                per_scores[c].extend(ps.tolist())
                continue
            iou = box_iou(pb, tb)
            matched = torch.zeros(tb.shape[0], dtype=torch.bool)
            for i in range(pb.shape[0]):
                j = iou[i].argmax().item()
                if iou[i, j] >= iou_thresh and not matched[j]:
                    per_tp[c].append(1); per_fp[c].append(0); matched[j] = True
                else:
                    per_tp[c].append(0); per_fp[c].append(1)
                per_scores[c].append(ps[i].item())

    aps = {}
    for c in range(num_classes):
        n = per_gt.get(c, 0)
        if n == 0 or len(per_scores[c]) == 0:
            aps[c] = 0.0
            continue
        scores = torch.tensor(per_scores[c])
        tp = torch.tensor(per_tp[c], dtype=torch.float32)
        fp = torch.tensor(per_fp[c], dtype=torch.float32)
        order = scores.argsort(descending=True)
        tp = tp[order].cumsum(0); fp = fp[order].cumsum(0)
        rec  = tp / n
        prec = tp / (tp + fp).clamp(min=1e-7)
        ap = 0.0
        for t in torch.linspace(0, 1, 101):
            m = rec >= t
            ap += (prec[m].max().item() if m.any() else 0.0) / 101.0
        aps[c] = ap
    mean_ap = sum(aps.values()) / max(len(aps), 1)
    return mean_ap, aps
