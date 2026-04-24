#!/usr/bin/env python3
"""Оценка обученного чекпоинта на валидационной выборке.

Использование:
    python scripts/val.py --ckpt assets/models/best.pt
"""
import sys
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from torch.utils.data import DataLoader

from src.dataset import YOLODataset, collate_fn
from src.model   import YOLO
from src.loss    import YOLOLoss
from src.utils   import load_config, get_device, load_checkpoint
from scripts.train import evaluate


def run(config_path: str, ckpt_path: str, split: str = "val"):
    cfg    = load_config(config_path)
    device = get_device()

    split_key = "val" if split == "val" else split
    ds = YOLODataset(
        split_file=cfg["data"][split_key],
        img_root=cfg["data"].get("img_dir"),
        img_size=cfg["data"]["imgsz"], augment=False,
    )
    loader = DataLoader(ds, batch_size=cfg["data"]["batch_size"], shuffle=False,
                         num_workers=cfg["data"]["workers"], collate_fn=collate_fn)
    print(f"сплит={split}   размер={len(ds)}")

    model = YOLO(num_classes=cfg["model"]["num_classes"],
                  pretrained=False, stride=cfg["model"]["stride"]).to(device)
    load_checkpoint(ckpt_path, model, device=str(device))
    criterion = YOLOLoss(num_classes=cfg["model"]["num_classes"], **cfg["loss"]).to(device)

    metrics = evaluate(model, criterion, loader, device, cfg,
                        amp=bool(cfg["device"]["amp"]))

    print(f"\nРезультаты на {split}:")
    print(f"  mAP50: {metrics['mAP50']:.4f}")
    for k in ("total", "obj", "bbox", "cls"):
        print(f"  {k}: {metrics[k]:.4f}")
    print("  AP по классам:")
    for c, ap in metrics["AP_per_class"].items():
        name = cfg.get("names", {}).get(c, str(c))
        print(f"    {c:>2}  {name:<15}  {ap:.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--ckpt",   default="assets/models/best.pt")
    args = p.parse_args()
    run(args.config, args.ckpt, split="val")
