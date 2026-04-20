#!/usr/bin/env python3
"""Training script for airplane detection"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.model import AirplaneDetector
from src.utils import load_config, setup_directories, get_device
import argparse


def main(args):
    # Load configuration
    config = load_config(args.config)
    
    # Setup directories
    dirs = setup_directories()
    
    # Initialize model
    print("🚀 Initializing model...")
    detector = AirplaneDetector(
        model_name=config['model']['name'],
        pretrained=config['model']['pretrained']
    )
    
    print(f"📊 Device: {detector.device}")
    
    # Train
    print("📚 Starting training...")
    results = detector.train(
        data=args.dataset_config,
        imgsz=config['data']['imgsz'],
        epochs=config['training']['epochs'],
        batch=config['data']['batch_size'],
        workers=config['data']['workers'],
        optimizer=config['training']['optimizer'],
        lr0=config['training']['lr0'],
        weight_decay=config['training']['weight_decay'],
        patience=config['training']['patience'],
        augment=config['augment']['enabled'],
        mosaic=config['augment']['mosaic'],
        hsv_h=config['augment']['hsv_h'],
        hsv_s=config['augment']['hsv_s'],
        hsv_v=config['augment']['hsv_v'],
        amp=config['device']['amp'],
        project=str(dirs['models']),
        name="airplane_detection",
        exist_ok=True
    )
    
    print("✅ Training completed!")
    print(f"📈 Best metrics: mAP50={results.results_dict.get('metrics/mAP50', 0):.4f}")
    
    # Save model
    model_path = dirs['models'] / "airplane_detection" / "weights" / "best.pt"
    print(f"💾 Model saved to: {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train airplane detection model")
    parser.add_argument("--config", type=str, default="configs/training.yaml", 
                       help="Path to training config")
    parser.add_argument("--dataset-config", type=str, default="configs/dataset.yaml",
                       help="Path to dataset config")
    
    args = parser.parse_args()
    main(args)