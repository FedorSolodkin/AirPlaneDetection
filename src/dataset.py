"""Датасет в формате YOLO: один .jpg на изображение + один .txt на изображение
с записями `cls cx cy w h` (нормализованные координаты, центральная форма).

Поддерживает два режима:
  1. split_file — путь к txt-файлу со списком путей к изображениям (train.txt / validation.txt / test.txt)
  2. img_dir + label_dir — папки с изображениями и метками (классическая структура YOLO)
"""
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

from .transform import letterbox, reproject_labels, augment as aug_fn, to_tensor


class YOLODataset(Dataset):
    def __init__(self, img_dir=None, label_dir=None,
                 split_file=None,
                 img_size: int = 640, augment: bool = False):
        """
        Параметры
        ----------
        split_file : str | Path | None
            Путь к txt-файлу со списком путей к изображениям.
            Метки ищутся рядом с изображением (тот же каталог, расширение .txt).
            Если задан — img_dir и label_dir игнорируются.
        img_dir : str | Path | None
            Папка с изображениями (классический режим).
        label_dir : str | Path | None
            Папка с метками (классический режим).
        """
        self.img_size = int(img_size)
        self.augment  = bool(augment)

        if split_file is not None:
            # Режим split-файла: читаем список путей из txt
            split_file = Path(split_file)
            lines = split_file.read_text().splitlines()
            # Корень проекта: data/hrplanes/train.txt -> data/hrplanes -> data -> project_root
            project_root = Path(split_file).resolve().parent.parent.parent
            self.paths = []
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                # Если путь относительный — резолвим от корня проекта
                p = Path(line) if Path(line).is_absolute() else project_root / line
                # Метка лежит рядом с изображением, расширение .txt
                lbl = p.with_suffix(".txt")
                if p.exists() and lbl.exists():
                    self.paths.append(p)
            self._label_dir = None  # метки ищем рядом с изображением
        else:
            # Классический режим: img_dir + label_dir
            self.img_dir   = Path(img_dir)
            self._label_dir = Path(label_dir)
            exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
            paths = []
            for e in exts:
                paths.extend(self.img_dir.glob(e))
            self.paths = sorted(
                p for p in paths if (self._label_dir / (p.stem + ".txt")).exists()
            )

    def _label_path(self, img_path: Path) -> Path:
        """Возвращает путь к файлу меток для данного изображения."""
        if self._label_dir is None:
            # Метка лежит рядом с изображением
            return img_path.with_suffix(".txt")
        return self._label_dir / (img_path.stem + ".txt")

    def __len__(self):
        return len(self.paths)

    @staticmethod
    def _load_labels(path: Path) -> np.ndarray:
        if not path.exists() or path.stat().st_size == 0:
            return np.zeros((0, 5), dtype=np.float32)
        arr = np.loadtxt(str(path), dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return arr.astype(np.float32, copy=False)

    def __getitem__(self, idx):
        p = self.paths[idx]
        img = Image.open(p).convert("RGB")
        labels = self._load_labels(self._label_path(p))

        canvas, scale, pad_x, pad_y, ow, oh = letterbox(img, self.img_size)
        labels = reproject_labels(labels, scale, pad_x, pad_y, ow, oh, self.img_size)

        if self.augment:
            canvas, labels = aug_fn(canvas, labels)

        if len(labels):
            keep = (labels[:, 3] > 1e-4) & (labels[:, 4] > 1e-4)
            labels = labels[keep]

        x = to_tensor(canvas)
        target = {
            "labels":   torch.as_tensor(labels[:, 0] if len(labels) else np.zeros((0,)),
                                         dtype=torch.int64),
            "boxes":    torch.as_tensor(labels[:, 1:5] if len(labels) else np.zeros((0, 4)),
                                         dtype=torch.float32),
            "image_id": torch.tensor([idx]),
        }
        return x, target


def collate_fn(batch):
    imgs = torch.stack([b[0] for b in batch], dim=0)
    targets = [b[1] for b in batch]
    return imgs, targets
