"""Конфиг, устройство, чекпоинты и всё математика боксов, используемая в проекте."""
import math
from pathlib import Path
import yaml
import torch


# ============================================================
# Конфиг / Ввод-вывод
# ============================================================

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def setup_dirs(base: str = "."):
    for sub in ("assets/logs", "assets/models", "assets/results"):
        Path(base, sub).mkdir(parents=True, exist_ok=True)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    print("CPU only")
    return torch.device("cpu")


def save_checkpoint(path, model, optimizer=None, epoch: int = 0, best_map: float = 0.0):
    state = {"model": model.state_dict(), "epoch": epoch, "best_map": best_map}
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    torch.save(state, path)


def load_checkpoint(path, model, optimizer=None, device: str = "cpu"):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt.get("epoch", 0), ckpt.get("best_map", 0.0)


# ============================================================
# Математика боксов
# ============================================================

def box_cxcywh_to_xyxy(x: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = x.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


def box_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Попарный IoU между боксами в формате xyxy. a: [N, 4], b: [M, 4] -> [N, M]."""
    area_a = (a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0)
    area_b = (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)
    lt = torch.max(a[:, None, :2], b[None, :, :2])
    rb = torch.min(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / union.clamp(min=1e-7)


def ciou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Поэлементный CIoU между двумя наборами боксов [N, 4] в формате xyxy. Возвращает [N]."""
    # IoU
    area_a = (a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0)
    area_b = (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)
    lt = torch.max(a[:, :2], b[:, :2])
    rb = torch.min(a[:, 2:], b[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = area_a + area_b - inter
    iou = inter / union.clamp(min=1e-7)

    # Диагональ наименьшего описывающего прямоугольника
    cw = torch.max(a[:, 2], b[:, 2]) - torch.min(a[:, 0], b[:, 0])
    ch = torch.max(a[:, 3], b[:, 3]) - torch.min(a[:, 1], b[:, 1])
    c2 = cw * cw + ch * ch + 1e-7

    # Расстояние между центрами
    cx_a = (a[:, 0] + a[:, 2]) / 2; cy_a = (a[:, 1] + a[:, 3]) / 2
    cx_b = (b[:, 0] + b[:, 2]) / 2; cy_b = (b[:, 1] + b[:, 3]) / 2
    rho2 = (cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2

    # Слагаемое согласованности соотношения сторон
    w_a = (a[:, 2] - a[:, 0]).clamp(min=1e-7)
    h_a = (a[:, 3] - a[:, 1]).clamp(min=1e-7)
    w_b = (b[:, 2] - b[:, 0]).clamp(min=1e-7)
    h_b = (b[:, 3] - b[:, 1]).clamp(min=1e-7)
    v = (4 / (math.pi ** 2)) * (torch.atan(w_b / h_b) - torch.atan(w_a / h_a)) ** 2
    with torch.no_grad():
        alpha = v / (1 - iou + v + 1e-7)
    return iou - rho2 / c2 - alpha * v


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_thresh: float = 0.5) -> torch.Tensor:
    """Жадное подавление немаксимумов (NMS). Возвращает индексы оставшихся боксов (отсортированы по score)."""
    if boxes.numel() == 0:
        return torch.empty(0, dtype=torch.int64, device=boxes.device)
    order = scores.argsort(descending=True)
    keep = []
    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)
        if order.numel() == 1:
            break
        iou = box_iou(boxes[i:i + 1], boxes[order[1:]])[0]
        order = order[1:][iou < iou_thresh]
    return torch.as_tensor(keep, dtype=torch.int64, device=boxes.device)
