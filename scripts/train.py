#!/usr/bin/env python3
"""Обучение YOLO-детектора самолётов.

Использование:
    python scripts/train.py --config configs/config.yaml
"""
import sys
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from src.dataset import YOLODataset, collate_fn
from src.model   import YOLO
from src.loss    import YOLOLoss
from src.metrics import decode_predictions, compute_map50
from src.utils   import (
    load_config, get_device, setup_dirs,
    save_checkpoint, box_cxcywh_to_xyxy,
)


# ------------------------------------------------------------------
# Построители компонентов
# ------------------------------------------------------------------
def build_loaders(cfg):
    sz       = cfg["data"]["imgsz"]
    img_root = cfg["data"].get("img_dir")  # реальная папка с изображениями
    train_ds = YOLODataset(
        split_file=cfg["data"]["train"],
        img_root=img_root,
        img_size=sz, augment=True,
    )
    val_ds = YOLODataset(
        split_file=cfg["data"]["val"],
        img_root=img_root,
        img_size=sz, augment=False,
    )
    train_loader = DataLoader(
        train_ds, batch_size=cfg["data"]["batch_size"], shuffle=True, drop_last=True,
        num_workers=cfg["data"]["workers"], collate_fn=collate_fn, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["data"]["batch_size"], shuffle=False, drop_last=False,
        num_workers=cfg["data"]["workers"], collate_fn=collate_fn, pin_memory=True,
    )
    return train_loader, val_loader


def build_optimizer(model, cfg):
    lr_head = float(cfg["training"]["lr0"])
    lr_bb   = float(cfg["training"]["lr_backbone"])
    wd      = float(cfg["training"]["weight_decay"])
    bb, other = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (bb if n.startswith("backbone") else other).append(p)
    return torch.optim.AdamW(
        [{"params": other, "lr": lr_head},
         {"params": bb,    "lr": lr_bb}],
        weight_decay=wd,
    )


# ------------------------------------------------------------------
# Циклы обучения и валидации
# ------------------------------------------------------------------
def train_one_epoch(model, criterion, optim, loader, device, scaler,
                     clip_max: float, amp: bool, epoch: int):
    model.train()
    running = {"total": 0.0, "obj": 0.0, "bbox": 0.0, "cls": 0.0}
    n = 0
    for step, (imgs, targets) in enumerate(loader):
        imgs     = imgs.to(device, non_blocking=True)
        targets  = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optim.zero_grad(set_to_none=True)
        with autocast(enabled=amp):
            out = model(imgs)
            losses = criterion(out, targets)

        scaler.scale(losses["total"]).backward()
        if clip_max > 0:
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max)
        scaler.step(optim)
        scaler.update()

        bs = imgs.size(0)
        for k in running:
            running[k] += losses[k].item() * bs
        n += bs
        if step % 20 == 0:
            print(f"  [ep {epoch} {step:4d}/{len(loader)}] "
                  f"total={losses['total'].item():.3f} "
                  f"obj={losses['obj'].item():.3f} "
                  f"bbox={losses['bbox'].item():.3f} "
                  f"cls={losses['cls'].item():.3f}")
    return {k: v / max(n, 1) for k, v in running.items()}


@torch.no_grad()
def evaluate(model, criterion, loader, device, cfg, amp: bool):
    model.eval()
    running = {"total": 0.0, "obj": 0.0, "bbox": 0.0, "cls": 0.0}
    n = 0
    all_preds, all_targets = [], []
    img_size = cfg["data"]["imgsz"]
    num_classes = cfg["model"]["num_classes"]

    for imgs, targets in loader:
        imgs      = imgs.to(device, non_blocking=True)
        targets_d = [{k: v.to(device) for k, v in t.items()} for t in targets]
        with autocast(enabled=amp):
            out = model(imgs)
            losses = criterion(out, targets_d)

        bs = imgs.size(0)
        for k in running:
            running[k] += losses[k].item() * bs
        n += bs

        preds = decode_predictions(
            out, img_size=img_size,
            conf_thresh=cfg["eval"]["conf_thresh"],
            top_k=cfg["eval"]["top_k"],
        )
        for b in range(bs):
            all_preds.append({k: v.cpu() for k, v in preds[b].items()})
            gt_xyxy = box_cxcywh_to_xyxy(targets[b]["boxes"]) * img_size
            all_targets.append({"boxes": gt_xyxy.cpu(),
                                 "labels": targets[b]["labels"].cpu()})

    metrics = {k: v / max(n, 1) for k, v in running.items()}
    map50, per_cls = compute_map50(all_preds, all_targets, num_classes=num_classes)
    metrics["mAP50"] = map50
    metrics["AP_per_class"] = per_cls
    return metrics


# ------------------------------------------------------------------
# Точка входа
# ------------------------------------------------------------------
def main(args):
    cfg = load_config(args.config)
    setup_dirs()
    device = get_device()

    train_loader, val_loader = build_loaders(cfg)
    print(f"train: {len(train_loader.dataset)}   val: {len(val_loader.dataset)}")

    model = YOLO(
        num_classes=cfg["model"]["num_classes"],
        pretrained =cfg["model"]["pretrained"],
        stride     =cfg["model"]["stride"],
    ).to(device)
    print(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    criterion = YOLOLoss(num_classes=cfg["model"]["num_classes"], **cfg["loss"]).to(device)
    optim     = build_optimizer(model, cfg)
    epochs    = int(cfg["training"]["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)
    amp       = bool(cfg["device"]["amp"])
    scaler    = GradScaler(enabled=amp)
    clip_max  = float(cfg["training"]["clip_max_norm"])

    ckpt_dir = Path("assets/models")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Папка для чекпоинтов каждой эпохи — на случай срыва обучения
    epoch_dir = ckpt_dir / "epochs"
    epoch_dir.mkdir(parents=True, exist_ok=True)

    best_map = 0.0
    for epoch in range(1, epochs + 1):
        print(f"\n=== Epoch {epoch}/{epochs}   "
              f"lr_head={optim.param_groups[0]['lr']:.2e}   "
              f"lr_bb={optim.param_groups[1]['lr']:.2e} ===")
        t = train_one_epoch(model, criterion, optim, train_loader, device,
                             scaler, clip_max=clip_max, amp=amp, epoch=epoch)
        v = evaluate(model, criterion, val_loader, device, cfg, amp=amp)
        scheduler.step()

        print("  train:", {k: round(v_, 3) for k, v_ in t.items()})
        print(f"  val:   mAP50={v['mAP50']:.4f}   "
              f"total={v['total']:.3f}  obj={v['obj']:.3f}  "
              f"bbox={v['bbox']:.3f}  cls={v['cls']:.3f}")

        # Сохраняем last.pt и чекпоинт текущей эпохи
        save_checkpoint(ckpt_dir / "last.pt", model, optim, epoch, best_map)
        save_checkpoint(epoch_dir / f"epoch_{epoch:03d}.pt", model, optim, epoch, best_map)

        if v["mAP50"] > best_map:
            best_map = v["mAP50"]
            save_checkpoint(ckpt_dir / "best.pt", model, optim, epoch, best_map)
            print(f"  💾 новый лучший mAP50={best_map:.4f}")

    print(f"\nГотово. Лучший mAP50={best_map:.4f}. Веса в {ckpt_dir}.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/config.yaml")
    main(p.parse_args())
