import urllib.request
import os
import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

# 10 realistic diverse non-soil photos and synthesized realistic photo textures
test_images = {
    "person_in_suit": "https://picsum.photos/id/1005/400/400", # Man portrait/person
    "indoor_living_room": "https://picsum.photos/id/1068/400/400", # Interior room
    "red_car_automobile": "https://picsum.photos/id/1071/400/400", # Vehicle
    "domestic_cat_pet": "https://picsum.photos/id/40/400/400", # Cat/animal
    "office_desk_laptop": "https://picsum.photos/id/0/400/400", # Laptop/tech
    "city_architecture": "https://picsum.photos/id/1015/400/400", # Buildings
    "coffee_cup_food": "https://picsum.photos/id/30/400/400", # Object/food
    "wooden_furniture": "https://picsum.photos/id/175/400/400", # Wood
    "fabric_clothing": "https://picsum.photos/id/1025/400/400", # Clothing/dog
    "smartphone_device": "https://picsum.photos/id/160/400/400" # Phone/device
}

os.makedirs('dataset/sanity_check_real_non_soil', exist_ok=True)
model = tf.keras.models.load_model('soil_binary_detector.keras', safe_mode=False)

print("=======================================================")
print("--- STAGE 1 REAL NON-SOIL IMAGE CONFIDENCE TEST ---")
print("=======================================================\n")

opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)')]
urllib.request.install_opener(opener)

for name, url in test_images.items():
    try:
        path = os.path.join('dataset/sanity_check_real_non_soil', f'{name}.jpg')
        if not os.path.exists(path) or os.path.getsize(path) < 1000:
            urllib.request.urlretrieve(url, path)
            
        img = Image.open(path).convert('RGB').resize((224, 224))
        arr = np.expand_dims(np.array(img), axis=0)
        prob = float(model.predict(arr, verbose=0)[0][0])
        status = "PASSED (REJECTED)" if prob < 0.75 else "FAILED (FALSE POSITIVE)"
        print(f"File: {name:<22} | Soil Probability: {prob*100:6.2f}% | Verdict: {status}")
    except Exception as e:
        print(f"File: {name:<22} | Error downloading/testing: {e}")
