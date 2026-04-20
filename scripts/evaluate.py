#!/usr/bin/env python3
"""Evaluation script"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.model import AirplaneDetector
from src.utils import load_config
import argparse


def main(args):
    config = load_config("configs/training.yaml")
    
    print("🔍 Loading model...")
    detector = AirplaneDetector()
    detector.load(args.model_path)
    
    print("📊 Evaluating...")
    metrics = detector.val(
        data=args.dataset_config,
        imgsz=config['data']['imgsz'],
        batch=config['data']['batch_size']
    )
    
    print("\n📈 Results:")
    print(f"  mAP50:    {metrics.results_dict.get('metrics/mAP50', 0):.4f}")
    print(f"  mAP50-95: {metrics.results_dict.get('metrics/mAP50-95', 0):.4f}")
    print(f"  Precision: {metrics.results_dict.get('metrics/precision', 0):.4f}")
    print(f"  Recall:   {metrics.results_dict.get('metrics/recall', 0):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, 
                       help="Path to model weights")
    parser.add_argument("--dataset-config", type=str, default="configs/dataset.yaml")
    
    args = parser.parse_args()
    main(args)