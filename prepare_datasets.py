"""
prepare_datasets.py

Final version: NO external AI-image datasets, NO EuroSAT, only CIFAR-10.

We build two binary datasets on disk:

1) MIDJOURNEY replacement:
   - REAL = CIFAR-10 images (original)
   - FAKE = "synthetic" CIFAR-10 images created by heavy transformations
            (inversion, solarize, posterize, blur, strong noise etc.)

2) STYLEGAN_SAT replacement:
   - REAL = CIFAR-10 images (original)
   - FAKE = "synthetic satellite-like" CIFAR-10 images created by a
            different set of heavy transforms (contrast, color shifts,
            sharpening, false-color-like effects, mild noise).

Folder structure:

data/
  MIDJOURNEY/
    train/REAL/
    train/FAKE/
    test/REAL/
    test/FAKE/
  STYLEGAN_SAT/
    train/REAL/
    train/FAKE/
    test/REAL/
    test/FAKE/

Requirements:
    pip install datasets pillow numpy
"""

import os
import random
import argparse
from typing import List, Tuple

from datasets import load_dataset
from PIL import Image, ImageOps, ImageFilter
import numpy as np


# ------------------------------------------------------
# Helpers
# ------------------------------------------------------

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_pil(img: Image.Image, path: str, size: int):
    img = img.convert("RGB")
    if size is not None:
        img = img.resize((size, size), Image.BILINEAR)
    img.save(path, "JPEG", quality=90)


def split_indices(n: int, ratio: float = 0.8, seed: int = 42) -> Tuple[list, list]:
    ids = list(range(n))
    random.Random(seed).shuffle(ids)
    k = int(n * ratio)
    return ids[:k], ids[k:]


# ------------------------------------------------------
# Synthetic "FAKE" image generators
# ------------------------------------------------------

def random_cifar_style_transform(img: Image.Image) -> Image.Image:
    """
    Heavy transforms to simulate an 'AI-ish' alternative distribution.
    Used for MIDJOURNEY FAKE images.
    """
    img = img.convert("RGB")

    # Random color inversion
    if random.random() < 0.5:
        img = ImageOps.invert(img)

    # Random solarize
    if random.random() < 0.5:
        img = ImageOps.solarize(img, threshold=random.randint(64, 192))

    # Random posterize
    if random.random() < 0.5:
        bits = random.choice([1, 2, 3, 4])
        img = ImageOps.posterize(img, bits)

    # Random blur
    if random.random() < 0.5:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 2.5)))

    # Random noise overlay
    if random.random() < 0.5:
        arr = np.array(img).astype(np.float32)
        noise = np.random.normal(0, 30, arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)

    return img


def random_satellite_style_transform(img: Image.Image) -> Image.Image:
    """
    Different heavy transforms intended to look more 'satellite-ish':
    strong contrast, color remapping, sharpening, mild noise, etc.
    Used for STYLEGAN_SAT FAKE images.
    """
    img = img.convert("RGB")

    # Convert to HSV-ish effect via posterize + solarize
    if random.random() < 0.5:
        img = ImageOps.posterize(img, random.choice([2, 3, 4]))
    if random.random() < 0.5:
        img = ImageOps.solarize(img, threshold=random.randint(64, 192))

    # Equalize or autocontrast to mimic satellite processing
    if random.random() < 0.5:
        img = ImageOps.equalize(img)
    if random.random() < 0.5:
        img = ImageOps.autocontrast(img, cutoff=random.randint(1, 10))

    # Sharpen edges
    if random.random() < 0.5:
        img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))

    # Mild blur occasionally
    if random.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))

    # Mild noise
    if random.random() < 0.5:
        arr = np.array(img).astype(np.float32)
        noise = np.random.normal(0, 20, arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)

    return img


# ------------------------------------------------------
# MIDJOURNEY replacement
# ------------------------------------------------------

def build_midjourney(base: str, n_real: int, n_fake: int, size: int, seed: int):
    root = os.path.join(base, "MIDJOURNEY")
    print(f"\n=== Building MIDJOURNEY at {root} ===")

    for split in ["train", "test"]:
        for cls in ["REAL", "FAKE"]:
            ensure_dir(os.path.join(root, split, cls))

    print("Loading CIFAR-10 from HuggingFace (REAL)...")
    cifar = load_dataset("cifar10", split="train")   # 50,000 images

    if n_real > len(cifar):
        print(f"Requested n_real={n_real} > {len(cifar)}. Clipping to {len(cifar)}.")
        n_real = len(cifar)

    random.seed(seed)
    real_indices = random.sample(range(len(cifar)), n_real)

    if n_fake > n_real:
        print(f"Requested n_fake={n_fake} > n_real={n_real}. Clipping to n_real.")
        n_fake = n_real
    fake_indices = real_indices[:n_fake]

    real_train, real_test = split_indices(n_real, 0.8, seed)
    fake_train, fake_test = split_indices(n_fake, 0.8, seed + 1)

    print("Saving REAL images...")
    for i, idx in enumerate(real_indices):
        img = cifar[idx]["img"]
        split = "train" if i in real_train else "test"
        save_pil(img, os.path.join(root, split, "REAL", f"real_{i:06d}.jpg"), size)

    print("Saving FAKE (synthetic) images for MIDJOURNEY...")
    for i, idx in enumerate(fake_indices):
        img = cifar[idx]["img"]
        img_fake = random_cifar_style_transform(img)
        split = "train" if i in fake_train else "test"
        save_pil(img_fake, os.path.join(root, split, "FAKE", f"fake_{i:06d}.jpg"), size)

    print("MIDJOURNEY-like dataset built.")


# ------------------------------------------------------
# STYLEGAN_SAT replacement
# ------------------------------------------------------

def build_stylegan_sat(base: str, n_real: int, n_fake: int, size: int, seed: int):
    root = os.path.join(base, "STYLEGAN_SAT")
    print(f"\n=== Building STYLEGAN_SAT at {root} ===")

    for split in ["train", "test"]:
        for cls in ["REAL", "FAKE"]:
            ensure_dir(os.path.join(root, split, cls))

    print("Loading CIFAR-10 from HuggingFace (REAL for STYLEGAN_SAT)...")
    cifar = load_dataset("cifar10", split="train")   # 50,000 images

    if n_real > len(cifar):
        print(f"Requested n_real={n_real} > {len(cifar)}. Clipping to {len(cifar)}.")
        n_real = len(cifar)

    random.seed(seed)
    real_indices = random.sample(range(len(cifar)), n_real)

    if n_fake > n_real:
        print(f"Requested n_fake={n_fake} > n_real={n_real}. Clipping to n_real.")
        n_fake = n_real
    fake_indices = real_indices[:n_fake]

    real_train, real_test = split_indices(n_real, 0.8, seed + 10)
    fake_train, fake_test = split_indices(n_fake, 0.8, seed + 11)

    print("Saving REAL images for STYLEGAN_SAT...")
    for i, idx in enumerate(real_indices):
        img = cifar[idx]["img"]
        # For REAL we only minimally process: just resize when saving
        split = "train" if i in real_train else "test"
        save_pil(img, os.path.join(root, split, "REAL", f"real_sat_{i:06d}.jpg"), size)

    print("Saving FAKE (synthetic satellite-style) images...")
    for i, idx in enumerate(fake_indices):
        img = cifar[idx]["img"]
        img_fake = random_satellite_style_transform(img)
        split = "train" if i in fake_train else "test"
        save_pil(img_fake, os.path.join(root, split, "FAKE", f"fake_sat_{i:06d}.jpg"), size)

    print("STYLEGAN_SAT-like dataset built.")


# ------------------------------------------------------
# Main
# ------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--n_real", type=int, default=20000)
    parser.add_argument("--n_fake", type=int, default=20000)
    parser.add_argument("--img_size", type=int, default=64)
    args = parser.parse_args()

    ensure_dir(args.data_root)

    build_midjourney(args.data_root, args.n_real, args.n_fake, args.img_size, seed=42)
    build_stylegan_sat(args.data_root, args.n_real, args.n_fake, args.img_size, seed=42)

    print("\nALL DATASETS READY.")


if __name__ == "__main__":
    main()
