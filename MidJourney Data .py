import os
import random
from pathlib import Path
from typing import List

import torch
from torchvision import datasets
from torchvision.transforms import ToPILImage
from diffusers import StableDiffusionPipeline
from PIL import Image, ImageOps
from tqdm.auto import tqdm


# =========================
# CONFIG
# =========================

BASE_DIR = Path("data_midjourney_like")

RAW_FAKE_DIR = BASE_DIR / "raw" / "fake"
RAW_REAL_DIR = BASE_DIR / "raw" / "real"

PROC_FAKE_DIR = BASE_DIR / "processed_64x64" / "fake"
PROC_REAL_DIR = BASE_DIR / "processed_64x64" / "real"

AUG_FAKE_DIR = BASE_DIR / "augmented_64x64" / "fake"
AUG_REAL_DIR = BASE_DIR / "augmented_64x64" / "real"

FINAL_DIR = BASE_DIR / "final_splits"

NUM_FAKE = 12_500
NUM_REAL = 12_500
TARGET_SIZE = (64, 64)
SD_IMAGE_SIZE = 256  # generation resolution before downscaling
RANDOM_SEED = 42


# =========================
# PROMPTS FOR SYNTHETIC IMAGES
# =========================

PROMPT_LIST: List[str] = [
    # Animals
    "a photorealistic close-up portrait of a cat looking at the camera, high detail, natural light",
    "a dog running through a grassy field, shallow depth of field, realistic photo",
    "a lion resting on a rock at golden hour, wildlife photography",

    # Landscapes
    "a wide-angle landscape with snow-covered mountains and a lake, dramatic sky, photo-realistic",
    "a dense forest with sun rays passing through the trees, ultra realistic",
    "a desert scene with sand dunes and a clear blue sky, realistic textures",

    # City / architecture
    "a nighttime cityscape with skyscrapers and car light trails, long exposure photography",
    "a European old town street with cobblestone and cafes, realistic daylight photo",
    "a modern glass office building reflecting the sunset, realistic photo",

    # People / portraits
    "a professional portrait of a young woman in studio lighting, 50mm lens, realistic skin",
    "a candid street photo of a man crossing the road in a busy city, natural ambient lighting",

    # Objects / still life
    "a realistic photograph of a cup of coffee on a wooden table, top-down view",
    "a pair of sneakers on a white background, studio product photography",
    "a bowl of fresh fruits on a kitchen counter, natural window light",

    # Interiors
    "a cozy living room with a sofa, coffee table and bookshelf, natural warm lighting, realistic",
    "a minimalist bedroom with a bed, nightstand and lamp, modern interior photography",

    # Vehicles
    "a red sports car on a mountain road at sunset, motion blur, realistic photography",
    "a passenger airplane flying above the clouds, aerial photography",

    # Food
    "a realistic close-up of a cheeseburger with fries, fast food photography",
    "a plate of sushi on a wooden table, restaurant lighting, realistic photo",
]


# =========================
# STEP 1: INIT STABLE DIFFUSION (HF MODEL)
# =========================

def init_sd_pipeline():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    model_path = Path("models/dreamshaper_8.safetensors")

    if not model_path.exists():
        raise FileNotFoundError(
            f"[ERROR] Could not find the model file at {model_path}\n"
            f"Make sure dreamshaper_8.safetensors is placed inside the 'models' folder."
        )

    print(f"[INFO] Loading local model from: {model_path}")

    pipe = StableDiffusionPipeline.from_single_file(
        str(model_path),
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)

    if device == "cuda":
        pipe.enable_attention_slicing()
        pipe.enable_vae_tiling()

    return pipe, device



# =========================
# STEP 2: GENERATE FAKE IMAGES
# =========================

def generate_fake_images(num_images: int = NUM_FAKE):
    RAW_FAKE_DIR.mkdir(parents=True, exist_ok=True)

    existing = list(RAW_FAKE_DIR.glob("fake_*.png"))
    start_idx = len(existing)
    remaining = num_images - start_idx

    if remaining <= 0:
        print(f"[INFO] Already have {len(existing)} fake images, skipping generation.")
        return

    print(f"[INFO] Need to generate {remaining} more fake images.")
    pipe, device = init_sd_pipeline()
    num_prompts = len(PROMPT_LIST)

    for i in tqdm(range(remaining), desc="Generating fake images"):
        global_idx = start_idx + i
        prompt = PROMPT_LIST[global_idx % num_prompts]

        if device == "cuda":
            with torch.autocast(device_type="cuda"):
                image = pipe(
                    prompt,
                    num_inference_steps=20,
                    guidance_scale=7.0,
                    height=SD_IMAGE_SIZE,
                    width=SD_IMAGE_SIZE,
                ).images[0]
        else:
            image = pipe(
                prompt,
                num_inference_steps=20,
                guidance_scale=7.0,
                height=SD_IMAGE_SIZE,
                width=SD_IMAGE_SIZE,
            ).images[0]

        image = image.convert("RGB")
        out_path = RAW_FAKE_DIR / f"fake_{global_idx:05d}.png"
        image.save(out_path)


# =========================
# STEP 3: DOWNLOAD REAL IMAGES (CIFAR-10)
# =========================

def build_real_images(num_images: int = NUM_REAL):
    RAW_REAL_DIR.mkdir(parents=True, exist_ok=True)

    existing = list(RAW_REAL_DIR.glob("real_*.png"))
    if len(existing) >= num_images:
        print(f"[INFO] Already have {len(existing)} real images, skipping CIFAR extraction.")
        return

    print("[INFO] Downloading CIFAR-10 (if not present) and extracting images...")
    cifar = datasets.CIFAR10(root=str(BASE_DIR / "cifar_data"), train=True, download=True)
    to_pil = ToPILImage()

    count = 0
    for idx, (img_tensor, label) in enumerate(tqdm(cifar, desc="Extracting real images")):
        if count >= num_images:
            break
        img = img_tensor
        img = img.convert("RGB")
        out_path = RAW_REAL_DIR / f"real_{count:05d}.png"
        img.save(out_path)
        count += 1

    print(f"[INFO] Saved {count} real images.")


# =========================
# STEP 4: RESIZE TO 64x64
# =========================

def resize_folder(in_dir: Path, out_dir: Path, size=TARGET_SIZE):
    out_dir.mkdir(parents=True, exist_ok=True)
    images = list(in_dir.glob("*.png"))

    for img_path in tqdm(images, desc=f"Resizing {in_dir.name}"):
        try:
            with Image.open(img_path) as im:
                im = im.convert("RGB")
                im = im.resize(size, Image.BICUBIC)
                out_path = out_dir / img_path.name
                im.save(out_path)
        except Exception as e:
            print(f"[WARN] Skipping {img_path}: {e}")


def resize_all_to_64():
    resize_folder(RAW_FAKE_DIR, PROC_FAKE_DIR)
    resize_folder(RAW_REAL_DIR, PROC_REAL_DIR)


# =========================
# STEP 5: AUGMENTATION
# =========================

def random_augmentation(im: Image.Image) -> Image.Image:
    ops = ["hflip", "vflip", "rot90", "rot180", "rot270"]
    op = random.choice(ops)
    if op == "hflip":
        return ImageOps.mirror(im)
    if op == "vflip":
        return ImageOps.flip(im)
    if op == "rot90":
        return im.rotate(90)
    if op == "rot180":
        return im.rotate(180)
    if op == "rot270":
        return im.rotate(270)
    return im


def augment_folder(in_dir: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    images = list(in_dir.glob("*.png"))

    for img_path in tqdm(images, desc=f"Augmenting {in_dir.name}"):
        try:
            with Image.open(img_path) as im:
                im = im.convert("RGB")
                aug = random_augmentation(im)
                out_path = out_dir / f"{img_path.stem}_aug.png"
                aug.save(out_path)
        except Exception as e:
            print(f"[WARN] Skipping {img_path}: {e}")


def augment_all():
    augment_folder(PROC_FAKE_DIR, AUG_FAKE_DIR)
    augment_folder(PROC_REAL_DIR, AUG_REAL_DIR)


# =========================
# STEP 6: BUILD TRAIN/VAL/TEST SPLITS
# =========================

def make_splits_for_class(proc_dir: Path, aug_dir: Path, out_base: Path,
                          split_ratios=(0.7, 0.15, 0.15), class_name="fake"):
    files = list(proc_dir.glob("*.png")) + list(aug_dir.glob("*.png"))
    files = sorted(files)
    random.seed(RANDOM_SEED)
    random.shuffle(files)

    n = len(files)
    n_train = int(split_ratios[0] * n)
    n_val = int(split_ratios[1] * n)
    n_test = n - n_train - n_val

    train_files = files[:n_train]
    val_files = files[n_train:n_train + n_val]
    test_files = files[n_train + n_val:]

    print(f"[INFO] {class_name}: total={n}, train={len(train_files)}, val={len(val_files)}, test={len(test_files)}")

    train_dir = out_base / "train" / class_name
    val_dir = out_base / "val" / class_name
    test_dir = out_base / "test" / class_name

    for d in [train_dir, val_dir, test_dir]:
        d.mkdir(parents=True, exist_ok=True)

    for src_list, dst_dir in [
        (train_files, train_dir),
        (val_files, val_dir),
        (test_files, test_dir),
    ]:
        for f in tqdm(src_list, desc=f"Copying {class_name} -> {dst_dir}"):
            dst_path = dst_dir / f.name
            if not dst_path.exists():
                with Image.open(f) as img:
                    img = img.convert("RGB")
                    img.save(dst_path)


def build_final_splits():
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    make_splits_for_class(PROC_FAKE_DIR, AUG_FAKE_DIR, FINAL_DIR, class_name="fake")
    make_splits_for_class(PROC_REAL_DIR, AUG_REAL_DIR, FINAL_DIR, class_name="real")


# =========================
# MAIN PIPELINE
# =========================

def main():
    os.makedirs(BASE_DIR, exist_ok=True)
    print(f"[INFO] Base dir: {BASE_DIR.resolve()}")

    # 1) Synthetic images (SD)
    generate_fake_images(NUM_FAKE)

    # 2) Real images (CIFAR-10)
    build_real_images(NUM_REAL)

    # 3) Resize all to 64x64
    resize_all_to_64()

    # 4) Augment all to double dataset size
    augment_all()

    # 5) Build final train/val/test splits
    build_final_splits()

    print("[INFO] Done. Final dataset is in:", FINAL_DIR.resolve())


if __name__ == "__main__":
    main()
