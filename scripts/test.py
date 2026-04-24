#!/usr/bin/env python3
"""Оценка обученного чекпоинта на тестовой выборке.

Использование:
    python scripts/test.py --ckpt assets/models/best.pt
"""
import sys
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from scripts.val import run


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--ckpt",   default="assets/models/best.pt")
    args = p.parse_args()
    run(args.config, args.ckpt, split="test")
