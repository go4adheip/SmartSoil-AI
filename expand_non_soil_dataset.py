import os
import urllib.request
import time
import shutil
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np

print("--- EXPANDING STAGE 1 NON-SOIL DATASET WITH REAL DIVERSE IMAGES ---")

non_soil_dir = os.path.join("dataset", "binary_dataset", "non_soil")
os.makedirs(non_soil_dir, exist_ok=True)

# Clear old synthetic images if needed
existing_files = [f for f in os.listdir(non_soil_dir) if f.startswith("non_soil_ac_unit") or f.startswith("non_soil_room")]
print(f"Removing {len(existing_files)} legacy synthetic files...")
for f in existing_files:
    try:
        os.remove(os.path.join(non_soil_dir, f))
    except Exception:
        pass

# 1. Download real photos from public image endpoints (picsum, unsplash) across 15 categories
categories = [
    # People & Suits & Clothing
    ("person_portrait", [1005, 1012, 1027, 1062, 1074]),
    ("suit_clothing", [1025, 824, 64, 331, 338]),
    # Vehicles & Transport
    ("car_vehicle", [1071, 111, 133, 183, 191]),
    # Animals & Pets
    ("animal_pet", [40, 169, 219, 237, 1024]),
    # Rooms & Furniture
    ("room_interior", [1068, 1069, 940, 950, 970]),
    ("furniture_wood", [175, 180, 201, 206, 211]),
    # Electronics & Devices
    ("laptop_tech", [0, 1, 2, 3, 4, 119, 160]),
    # Architecture & Buildings
    ("architecture_city", [1015, 1031, 1040, 1043, 1057]),
    # Food & Objects
    ("food_drink", [30, 42, 43, 102, 1060]),
    # Fabrics & Textiles
    ("fabric_pattern", [106, 108, 110, 115, 120]),
]

opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)')]
urllib.request.install_opener(opener)

count_downloaded = 0
base_ids = []
for cat, img_ids in categories:
    base_ids.extend([(cat, i) for i in img_ids])

# We also generate variations (crops, color shifts, rotations, flips) to create 1,200 real photographic non-soil images
print("Downloading base real photographic seeds...")
base_images = []

for idx, (cat, img_id) in enumerate(base_ids):
    url = f"https://picsum.photos/id/{img_id}/400/400"
    out_path = os.path.join(non_soil_dir, f"real_seed_{cat}_{idx}.jpg")
    try:
        if not os.path.exists(out_path):
            urllib.request.urlretrieve(url, out_path)
            time.sleep(0.1)
        if os.path.exists(out_path) and os.path.getsize(out_path) > 1000:
            img = Image.open(out_path).convert("RGB")
            base_images.append((cat, img))
            count_downloaded += 1
    except Exception as e:
        print(f"Failed to fetch {url}: {e}")

print(f"Successfully loaded {len(base_images)} base photographic non-soil seeds.")

# Augment seeds into 1,200 diverse non-soil training samples
print("Generating 1,200 augmented photographic non-soil images...")
total_augmented = 0

for cat, img in base_images:
    w, h = img.size
    for var in range(25):
        # Random crop
        cw, ch = int(w * np.random.uniform(0.5, 0.95)), int(h * np.random.uniform(0.5, 0.95))
        cx = np.random.randint(0, w - cw)
        cy = np.random.randint(0, h - ch)
        cropped = img.crop((cx, cy, cx + cw, cy + ch)).resize((224, 224))
        
        # Random flips & adjustments
        if np.random.rand() > 0.5:
            cropped = cropped.transpose(Image.FLIP_LEFT_RIGHT)
        if np.random.rand() > 0.5:
            cropped = cropped.transpose(Image.FLIP_TOP_BOTTOM)
            
        enhancer = ImageEnhance.Color(cropped)
        cropped = enhancer.enhance(np.random.uniform(0.6, 1.4))
        
        enhancer_b = ImageEnhance.Brightness(cropped)
        cropped = enhancer_b.enhance(np.random.uniform(0.7, 1.3))
        
        fname = f"real_non_soil_{cat}_{var}_{total_augmented}.jpg"
        cropped.save(os.path.join(non_soil_dir, fname), "JPEG", quality=90)
        total_augmented += 1

print(f"Total non-soil training images ready: {len(os.listdir(non_soil_dir))}")
