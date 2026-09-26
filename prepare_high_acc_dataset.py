import os
import shutil
import random
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

print("--- STEP 1: PREPARING CLEAN & BALANCED HELD-OUT DATASET SPLIT ---")

seed = 42
random.seed(seed)
np.random.seed(seed)

orig_dir = "Soil-Classification-Dataset-main/Orignal-Dataset"
output_base = "dataset/multiclass_split"

train_dir = os.path.join(output_base, "train")
val_dir = os.path.join(output_base, "val")
test_dir = os.path.join(output_base, "test")

# Re-create clean target directories
for d in [train_dir, val_dir, test_dir]:
    if os.path.exists(d):
        shutil.rmtree(d)
    os.makedirs(d, exist_ok=True)

classes = sorted(os.listdir(orig_dir))
print("Soil Classes found:", classes)

target_per_class = 350  # Target count per class after augmentations

for c in classes:
    class_src = os.path.join(orig_dir, c)
    if not os.path.isdir(class_src):
        continue
        
    os.makedirs(os.path.join(train_dir, c), exist_ok=True)
    os.makedirs(os.path.join(val_dir, c), exist_ok=True)
    os.makedirs(os.path.join(test_dir, c), exist_ok=True)
    
    # 1. Load and filter valid original images
    valid_files = []
    for fname in os.listdir(class_src):
        fpath = os.path.join(class_src, fname)
        if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
            try:
                with Image.open(fpath) as img:
                    img.verify()
                valid_files.append(fpath)
            except Exception:
                pass
                
    random.shuffle(valid_files)
    n_total = len(valid_files)
    
    # Split 70% train, 15% val, 15% test
    n_train = int(0.70 * n_total)
    n_val = int(0.15 * n_total)
    
    train_files = valid_files[:n_train]
    val_files = valid_files[n_train:n_train + n_val]
    test_files = valid_files[n_train + n_val:]
    
    print(f"Class: {c:<15} | Raw Total: {n_total:3d} | Train: {len(train_files):3d}, Val: {len(val_files):2d}, Test: {len(test_files):2d}")
    
    # Copy Validation and Test files directly (UNTOUCHED / HELD-OUT)
    for idx, fp in enumerate(val_files):
        shutil.copy(fp, os.path.join(val_dir, c, f"val_{idx}_{os.path.basename(fp)}"))
    for idx, fp in enumerate(test_files):
        shutil.copy(fp, os.path.join(test_dir, c, f"test_{idx}_{os.path.basename(fp)}"))
        
    # Copy Train base files
    for idx, fp in enumerate(train_files):
        shutil.copy(fp, os.path.join(train_dir, c, f"train_orig_{idx}_{os.path.basename(fp)}"))
        
    # Augment Training Set ONLY to balance classes up to ~300 training samples per class
    current_train_count = len(os.listdir(os.path.join(train_dir, c)))
    aug_needed = max(0, 300 - current_train_count)
    
    if aug_needed > 0 and len(train_files) > 0:
        for i in range(aug_needed):
            src_fp = random.choice(train_files)
            try:
                img = Image.open(src_fp).convert("RGB")
                w, h = img.size
                
                # Apply realistic phone photo augmentations (rotations, zoom, brightness/contrast, noise/blur)
                angle = random.uniform(-30, 30)
                img_aug = img.rotate(angle, expand=False, resample=Image.BILINEAR)
                
                if random.random() > 0.5:
                    img_aug = img_aug.transpose(Image.FLIP_LEFT_RIGHT)
                if random.random() > 0.5:
                    img_aug = img_aug.transpose(Image.FLIP_TOP_BOTTOM)
                    
                # Crop
                crop_factor = random.uniform(0.75, 0.95)
                cw, ch = int(w * crop_factor), int(h * crop_factor)
                cx = random.randint(0, w - cw)
                cy = random.randint(0, h - ch)
                img_aug = img_aug.crop((cx, cy, cx + cw, cy + ch)).resize((224, 224))
                
                # Brightness / Contrast jitter
                enh_c = ImageEnhance.Color(img_aug)
                img_aug = enh_c.enhance(random.uniform(0.7, 1.3))
                enh_b = ImageEnhance.Brightness(img_aug)
                img_aug = enh_b.enhance(random.uniform(0.7, 1.3))
                
                # Slight Gaussian blur to simulate low quality phone camera focus
                if random.random() > 0.6:
                    img_aug = img_aug.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.2)))
                    
                aug_fname = f"train_aug_{i}_{os.path.basename(src_fp)}"
                img_aug.save(os.path.join(train_dir, c, aug_fname), "JPEG", quality=90)
            except Exception:
                pass

print("\n--- Summary of Prepared Split ---")
for split, sdir in [("Train", train_dir), ("Val", val_dir), ("Test", test_dir)]:
    counts = {c: len(os.listdir(os.path.join(sdir, c))) for c in classes}
    print(f"{split:<6} set counts per class: {counts}")
