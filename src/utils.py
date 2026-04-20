import yaml
import torch
import matplotlib.pyplot as plt
from pathlib import Path


def load_config(config_path: str):
    """Load YAML configuration"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def setup_directories(base_dir: str = "."):
    """Create necessary directories"""
    dirs = {
        'logs': Path(base_dir) / "assets" / "logs",
        'models': Path(base_dir) / "assets" / "models",
        'results': Path(base_dir) / "assets" / "results"
    }
    
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return dirs


def plot_training_results(results_dir: str):
    """Plot training metrics"""
    results_path = Path(results_dir)
    
    if not results_path.exists():
        print("No results found")
        return
    
    # Plot results.png if exists
    results_img = results_path / "results.png"
    if results_img.exists():
        img = plt.imread(results_img)
        plt.figure(figsize=(15, 10))
        plt.imshow(img)
        plt.axis('off')
        plt.tight_layout()
        plt.show()


def get_device():
    """Get available device"""
    if torch.cuda.is_available():
        print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    else:
        print("⚠️ Using CPU")
        return torch.device("cpu")