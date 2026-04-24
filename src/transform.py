"""Преобразования изображений и меток: letterbox, аугментации, конвертация в тензор."""
import random
import numpy as np
import torch
from PIL import Image
import torchvision.transforms.functional as TF


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def letterbox(img: Image.Image, new_size: int):
    """Масштабирует изображение с сохранением пропорций и дополняет до квадрата размером `new_size`.
    Возвращает холст и параметры, необходимые для пересчёта координат в обоих направлениях.
    """
    w, h = img.size
    scale = new_size / max(w, h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    img = img.resize((nw, nh), Image.BILINEAR)
    pad_w, pad_h = new_size - nw, new_size - nh
    left, top = pad_w // 2, pad_h // 2
    canvas = Image.new("RGB", (new_size, new_size), (114, 114, 114))
    canvas.paste(img, (left, top))
    return canvas, scale, left, top, w, h


def reproject_labels(labels: np.ndarray, scale: float, pad_x: int, pad_y: int,
                      orig_w: int, orig_h: int, new_size: int) -> np.ndarray:
    """Пересчитывает метки из нормализованных координат оригинала в нормализованные координаты холста."""
    if len(labels) == 0:
        return labels
    cx = labels[:, 1] * orig_w * scale + pad_x
    cy = labels[:, 2] * orig_h * scale + pad_y
    bw = labels[:, 3] * orig_w * scale
    bh = labels[:, 4] * orig_h * scale
    labels[:, 1] = cx / new_size
    labels[:, 2] = cy / new_size
    labels[:, 3] = bw / new_size
    labels[:, 4] = bh / new_size
    labels[:, 1:5] = np.clip(labels[:, 1:5], 0.0, 1.0)
    return labels


def augment(img: Image.Image, labels: np.ndarray):
    """Простые аугментации: горизонтальный флип + изменение цвета. Метки синхронизируются с изображением."""
    if random.random() < 0.5:
        img = TF.hflip(img)
        if len(labels):
            labels[:, 1] = 1.0 - labels[:, 1]
    if random.random() < 0.7:
        img = TF.adjust_brightness(img, 0.8 + 0.4 * random.random())
        img = TF.adjust_contrast(img,   0.8 + 0.4 * random.random())
        img = TF.adjust_saturation(img, 0.8 + 0.4 * random.random())
        img = TF.adjust_hue(img, 0.1 * (random.random() - 0.5))
    return img, labels


def to_tensor(img: Image.Image) -> torch.Tensor:
    x = TF.to_tensor(img)
    return TF.normalize(x, IMAGENET_MEAN, IMAGENET_STD)
