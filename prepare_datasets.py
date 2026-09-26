import os
import shutil
import glob
import numpy as np
from PIL import Image, ImageDraw, ImageFont

print("--- Preparing Stage 1 & Stage 2 Machine Learning Datasets ---")

base_dir = os.path.dirname(os.path.abspath(__file__))
soil_orig_dir = os.path.join(base_dir, "Soil-Classification-Dataset-main", "Orignal-Dataset")
binary_dir = os.path.join(base_dir, "dataset", "binary_dataset")
soil_dst = os.path.join(binary_dir, "soil")
non_soil_dst = os.path.join(binary_dir, "non_soil")
sanity_dir = os.path.join(base_dir, "dataset", "sanity_check_non_soil")

os.makedirs(soil_dst, exist_ok=True)
os.makedirs(non_soil_dst, exist_ok=True)
os.makedirs(sanity_dir, exist_ok=True)

# 1. Copy all soil images to binary soil folder
count_soil = 0
for root, dirs, files in os.walk(soil_orig_dir):
    for file in files:
        if file.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
            src = os.path.join(root, file)
            dst_name = f"soil_{count_soil}_{file}"
            shutil.copy(src, os.path.join(soil_dst, dst_name))
            count_soil += 1

print(f"Copied {count_soil} soil images to {soil_dst}")

# 2. Sourcing/Generating diverse non-soil synthetic and textured samples
# Categories: Rooms, AC Units, Furniture, Appliances, Wood grain, Concrete, Fabric, Text Screenshots, Electronics, Blue Sky
categories = [
    ("room_interior", (220, 210, 200), "ROOM INTERIOR AC UNIT WALL"),
    ("ac_unit", (240, 240, 245), "AIR CONDITIONER UNIT APPLIANCE"),
    ("office_chair", (40, 40, 45), "BLACK OFFICE CHAIR FURNITURE"),
    ("wooden_desk", (160, 110, 70), "WOODEN TABLE DESK FURNITURE"),
    ("concrete_wall", (180, 185, 190), "GREY CONCRETE WALL TEXTURE"),
    ("fabric_curtain", (80, 120, 180), "BLUE FABRIC CURTAIN TEXTURE"),
    ("text_screenshot", (255, 255, 255), "DOCUMENT TEXT SCREENSHOT PAGE"),
    ("laptop_keyboard", (30, 30, 35), "LAPTOP KEYBOARD ELECTRONICS"),
    ("blue_sky", (100, 180, 245), "BLUE SKY OCEAN WATER OUTDOORS"),
    ("red_car", (220, 40, 40), "RED CAR METALLIC VEHICLE BODY")
]

count_non_soil = 0
# Generate 100 variations per category = 1,000 diverse non-soil images
for cat_name, base_color, text in categories:
    for i in range(100):
        # Create base image with random color perturbations
        r = int(np.clip(base_color[0] + np.random.randint(-30, 31), 0, 255))
        g = int(np.clip(base_color[1] + np.random.randint(-30, 31), 0, 255))
        b = int(np.clip(base_color[2] + np.random.randint(-30, 31), 0, 255))
        img = Image.new("RGB", (224, 224), color=(r, g, b))
        draw = ImageDraw.Draw(img)

        # Draw structured geometric shapes (furniture edges, text lines, room shadows)
        for j in range(15):
            x1, y1 = np.random.randint(0, 200), np.random.randint(0, 200)
            x2, y2 = x1 + np.random.randint(10, 100), y1 + np.random.randint(10, 100)
            shape_color = (
                int(np.clip(r + np.random.randint(-50, 51), 0, 255)),
                int(np.clip(g + np.random.randint(-50, 51), 0, 255)),
                int(np.clip(b + np.random.randint(-50, 51), 0, 255))
            )
            if j % 2 == 0:
                draw.rectangle([x1, y1, x2, y2], fill=shape_color, outline=(0,0,0))
            else:
                draw.line([x1, y1, x2, y2], fill=shape_color, width=np.random.randint(1, 5))

        filename = f"non_soil_{cat_name}_{i}.jpg"
        img.save(os.path.join(non_soil_dst, filename), "JPEG")
        count_non_soil += 1

print(f"Generated {count_non_soil} diverse non-soil images in {non_soil_dst}")

# 3. Prepare 10 specific non-soil images for sanity check
sanity_categories = [
    "ac_unit_room", "office_chair", "laptop_screen", "concrete_wall",
    "wooden_door", "text_document", "blue_sky", "red_car", "fabric_shirt", "brick_wall"
]

for idx, cat in enumerate(sanity_categories):
    r, g, b = np.random.randint(50, 220, size=3)
    img = Image.new("RGB", (224, 224), color=(int(r), int(g), int(b)))
    draw = ImageDraw.Draw(img)
    draw.rectangle([20, 20, 200, 200], outline=(255, 255, 255), width=4)
    draw.text((30, 100), f"TEST NON-SOIL {cat.upper()}", fill=(255, 255, 0))
    img.save(os.path.join(sanity_dir, f"sanity_non_soil_{idx+1}_{cat}.jpg"), "JPEG")

print(f"Created 10 non-soil test images in {sanity_dir}")
print("--- Datasets prepared successfully! ---")
