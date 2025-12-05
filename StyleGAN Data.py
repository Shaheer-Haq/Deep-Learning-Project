import os
import random
from pathlib import Path
from typing import List

import torch
from diffusers import StableDiffusionPipeline
from PIL import Image, ImageOps
from tqdm.auto import tqdm


# =========================
# CONFIG
# =========================

BASE_DIR = Path("data_stylegan_like")

EUROSAT_DIR = Path("EuroSAT")  # your real EuroSAT RGB folder

PROC_REAL_DIR = BASE_DIR / "real_128"
PROC_FAKE_DIR = BASE_DIR / "fake_128"

FINAL_DIR = BASE_DIR / "final_splits"

NUM_REAL_TARGET = 50_000
NUM_FAKE_TARGET = 50_000
TARGET_SIZE = (128, 128)
SD_IMAGE_SIZE = 256  # generate at 256, then downscale to 128
RANDOM_SEED = 42


# =========================
# SATELLITE PROMPTS FOR SYNTHETIC IMAGES
# =========================

SATELLITE_PROMPTS: List[str] = [
    # Urban
    "high-altitude satellite view of a dense urban city grid with roads and buildings, top-down, realistic colors",
    "satellite view of a suburban residential neighborhood with houses and streets, top-down, clear details",
    "satellite view of an industrial zone with warehouses, factories and parking lots, top-down",

    # Agriculture
    "satellite view of large agricultural fields divided into rectangular plots with different crops, top-down, realistic",
    "top-down satellite image of circular irrigation fields in a desert region, vivid contrast",
    "satellite view of mixed farmland with patchy fields and dirt roads, top-down",

    # Water / coasts
    "satellite view of a coastal city with beaches and ocean waves, top-down",
    "satellite image of a river meandering through green landscape, top-down, realistic",
    "top-down satellite view of a large lake surrounded by forests and fields",

    # Forests / nature
    "satellite view of dense forest with variations in tree color and texture, top-down",
    "satellite image of mountains with snow-covered peaks and rocky terrain, top-down",
    "high-altitude view of tundra landscape with patches of snow and rock, satellite style",

    # Deserts
    "satellite view of sand dunes in a desert region with soft shadows, top-down",
    "satellite image of rocky desert with sparse vegetation and dry riverbeds, top-down",

    # Infrastructure
    "top-down satellite view of a highway intersection with multiple lanes and cars, realistic",
    "satellite image of a major port with docks, ships and containers, top-down",
    "satellite view of an airport with runways, taxiways and terminals, top-down",

    # Mixed / misc
    "satellite image of a coastal wetland with marshes, channels and small islands, top-down",
    "satellite view of patchy clouds over land and ocean, realistic textures, top-down",
    "satellite view of mosaic of urban, agricultural and forest areas in one scene, top-down",
]


# =========================
# STEP 1: INIT STABLE DIFFUSION (LOCAL DREAMSHAPER)
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
# STEP 2: REAL IMAGES FROM EUROSAT (UPSCALE + AUGMENT TO 50K)
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


def build_real_images_from_eurosat(target_num: int = NUM_REAL_TARGET):
    PROC_REAL_DIR.mkdir(parents=True, exist_ok=True)

    if not EUROSAT_DIR.exists():
        raise FileNotFoundError(
            f"[ERROR] EuroSAT directory not found at {EUROSAT_DIR}. "
            f"Expected path: ./EuroSAT with class subfolders inside."
        )

    existing = list(PROC_REAL_DIR.glob("real_*.png"))
    if len(existing) >= target_num:
        print(f"[INFO] Already have {len(existing)} real images, skipping EuroSAT processing.")
        return

    print(f"[INFO] Scanning EuroSAT images in {EUROSAT_DIR.resolve()}")

    image_paths = []
    for class_dir in sorted(EUROSAT_DIR.iterdir()):
        if class_dir.is_dir():
            for img_path in class_dir.glob("*"):
                if img_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                    image_paths.append(img_path)

    if not image_paths:
        raise RuntimeError(f"[ERROR] No images found under {EUROSAT_DIR}")

    print(f"[INFO] Found {len(image_paths)} base EuroSAT images.")
    random.seed(RANDOM_SEED)
    random.shuffle(image_paths)

    # 1) First pass: resize + save all original EuroSAT images
    count = 0
    for img_path in tqdm(image_paths, desc="Saving base real images"):
        if count >= target_num:
            break
        try:
            with Image.open(img_path) as im:
                im = im.convert("RGB")
                im = im.resize(TARGET_SIZE, Image.BICUBIC)
                out_path = PROC_REAL_DIR / f"real_{count:05d}.png"
                im.save(out_path)
                count += 1
        except Exception as e:
            print(f"[WARN] Skipping {img_path}: {e}")

    # 2) If still below target_num, create augmented copies
    if count < target_num:
        print(f"[INFO] Need {target_num - count} more real images via augmentation.")
        existing_real = list(PROC_REAL_DIR.glob("real_*.png"))
        base_images = existing_real.copy()
        idx = count

        while idx < target_num:
            src = random.choice(base_images)
            try:
                with Image.open(src) as im:
                    im = im.convert("RGB")
                    aug = random_augmentation(im)
                    out_path = PROC_REAL_DIR / f"real_{idx:05d}.png"
                    aug.save(out_path)
                    idx += 1
            except Exception as e:
                print(f"[WARN] Augment skip {src}: {e}")

    final_count = len(list(PROC_REAL_DIR.glob("real_*.png")))
    print(f"[INFO] Final real image count: {final_count}")


# =========================
# STEP 3: GENERATE FAKE SATELLITE IMAGES (50K)
# =========================

def generate_fake_images(target_num: int = NUM_FAKE_TARGET):
    """
    Generate synthetic images with adaptive batch size:
    - Start from MAX_BATCH
    - On CUDA OOM: reduce batch size and retry
    - On long success streak: try to increase batch size again
    """
    PROC_FAKE_DIR.mkdir(parents=True, exist_ok=True)

    # ---------- CONFIG ----------
    MAX_BATCH = 3      # upper limit you want to try
    MIN_BATCH = 1      # never go below 1
    START_BATCH = 2    # initial batch size (good for 8GB)
    SUCCESS_STREAK_FOR_INCREASE = 10  # batches without OOM before trying to bump up
    # ----------------------------

    existing = list(PROC_FAKE_DIR.glob("fake_*.png"))
    start_idx = len(existing)
    remaining = target_num - start_idx

    if remaining <= 0:
        print(f"[INFO] Already have {len(existing)} fake images, skipping generation.")
        return

    print(f"[INFO] Need to generate {remaining} more fake images.")
    pipe, device = init_sd_pipeline()
    num_prompts = len(SATELLITE_PROMPTS)  # or PROMPT_LIST in data1.py

    current_batch = min(START_BATCH, MAX_BATCH)
    success_streak = 0

    pbar = tqdm(total=remaining, desc="Generating fake images (adaptive batch)")

    produced = 0
    global_idx = start_idx

    while produced < remaining:
        # clamp batch size to what’s left
        this_batch = min(current_batch, remaining - produced)

        # build prompts for this batch
        prompts = [
            SATELLITE_PROMPTS[(global_idx + j) % num_prompts]
            for j in range(this_batch)
        ]

        try:
            # --- do the actual generation ---
            if device == "cuda":
                with torch.autocast(device_type="cuda"):
                    out = pipe(
                        prompts,
                        num_inference_steps=20,
                        guidance_scale=7.0,
                        height=SD_IMAGE_SIZE,
                        width=SD_IMAGE_SIZE,
                    )
                    images = out.images
            else:
                out = pipe(
                    prompts,
                    num_inference_steps=20,
                    guidance_scale=7.0,
                    height=SD_IMAGE_SIZE,
                    width=SD_IMAGE_SIZE,
                )
                images = out.images

        except RuntimeError as e:
            msg = str(e).lower()
            # Handle CUDA OOM: reduce batch size and retry
            if "out of memory" in msg or "cuda" in msg and "memory" in msg:
                print(
                    f"\n[WARN] CUDA OOM at batch_size={current_batch}. "
                    f"Reducing batch size."
                )
                # clear CUDA cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # reduce batch size
                if current_batch > MIN_BATCH:
                    current_batch = max(MIN_BATCH, current_batch - 1)
                    print(f"[INFO] New batch size: {current_batch}")
                    # retry same iteration with smaller batch
                    continue
                else:
                    # even batch=1 fails → hard stop; user needs to lower steps/resolution
                    raise RuntimeError(
                        "[FATAL] CUDA OOM even at batch_size=1. "
                        "Reduce SD_IMAGE_SIZE or num_inference_steps."
                    ) from e
            else:
                # some other runtime error: re-raise
                raise

        # --- save images if generation succeeded ---
        for img in images:
            img = img.convert("RGB").resize(TARGET_SIZE, Image.BICUBIC)
            out_path = PROC_FAKE_DIR / f"fake_{global_idx:05d}.png"
            img.save(out_path)
            global_idx += 1
            produced += 1
            pbar.update(1)

        # successful batch
        success_streak += 1

        # if we've been stable for a while, try increasing batch size (up to MAX_BATCH)
        if success_streak >= SUCCESS_STREAK_FOR_INCREASE and current_batch < MAX_BATCH:
            trial_batch = current_batch + 1
            print(
                f"[INFO] {success_streak} successful batches at batch_size={current_batch}. "
                f"Trying to increase to {trial_batch}."
            )
            current_batch = trial_batch
            success_streak = 0  # reset streak for next level

        # optional: occasionally free cache to avoid fragmentation
        if torch.cuda.is_available() and produced % 500 == 0:
            torch.cuda.empty_cache()

    pbar.close()



# =========================
# STEP 4: BUILD TRAIN/VAL/TEST SPLITS
# =========================

def make_splits_for_class(src_dir: Path, out_base: Path,
                          split_ratios=(0.7, 0.15, 0.15), class_name="real"):
    files = sorted(src_dir.glob("*.png"))
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
    make_splits_for_class(PROC_REAL_DIR, FINAL_DIR, class_name="real")
    make_splits_for_class(PROC_FAKE_DIR, FINAL_DIR, class_name="fake")


# =========================
# MAIN PIPELINE
# =========================

def main():
    os.makedirs(BASE_DIR, exist_ok=True)
    print(f"[INFO] Base dir: {BASE_DIR.resolve()}")

    # 1) Real EuroSAT-based images to 50K
    build_real_images_from_eurosat(NUM_REAL_TARGET)

    # 2) Synthetic satellite images via DreamShaper
    generate_fake_images(NUM_FAKE_TARGET)

    # 3) Train/val/test split
    build_final_splits()

    print("[INFO] Done. Final dataset is in:", FINAL_DIR.resolve())


if __name__ == "__main__":
    main()
