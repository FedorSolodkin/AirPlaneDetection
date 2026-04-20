#!/usr/bin/env python3
"""Inference script"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.model import AirplaneDetector
import argparse


def main(args):
    print("🔍 Loading model...")
    detector = AirplaneDetector()
    
    if args.model_path:
        detector.load(args.model_path)
    
    print(f"📸 Running inference on: {args.source}")
    results = detector.predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        save=args.save
    )
    
    print(f"✅ Found {len(results[0].boxes)} objects")
    
    if args.save:
        results[0].save(filename=args.output)
        print(f"💾 Result saved to: {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=True, 
                       help="Path to image or folder")
    parser.add_argument("--model-path", type=str, default=None,
                       help="Path to custom model weights")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--save", action="store_true", default=True)
    parser.add_argument("--output", type=str, default="output.jpg")
    
    args = parser.parse_args()
    main(args)