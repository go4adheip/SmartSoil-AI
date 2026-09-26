import os
import tensorflow as tf
import numpy as np
from PIL import Image

print("=======================================================")
print("--- STAGE 1 SANITY CHECK: REAL NON-SOIL IMAGES TEST ---")
print("=======================================================")

model_path = "soil_binary_detector.keras"
if not os.path.exists(model_path):
    print("Error: soil_binary_detector.keras model file not found!")
    exit(1)

model = tf.keras.models.load_model(model_path, safe_mode=False)

sanity_dir = "dataset/sanity_check_real_non_soil"
test_files = sorted([f for f in os.listdir(sanity_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

total_tested = 0
correctly_rejected = 0

print(f"Testing {len(test_files)} real non-soil images against Stage 1 Soil Binary Detector...\n")

for f in test_files:
    img_path = os.path.join(sanity_dir, f)
    img_obj = Image.open(img_path).convert("RGB").resize((224, 224))
    img_arr = np.expand_dims(np.array(img_obj), axis=0)

    # Output sigmoid: Class 0: non_soil, Class 1: soil
    pred_prob = float(model.predict(img_arr, verbose=0)[0][0])
    is_soil = pred_prob >= 0.75  # 75% hard threshold

    total_tested += 1
    if not is_soil:
        correctly_rejected += 1
        status = "PASSED (CORRECTLY REJECTED)"
    else:
        status = "FAILED (FALSE POSITIVE)"

    print(f"File: {f:<25} | Soil Probability: {pred_prob*100:6.2f}% | Decision: {status}")

rejection_rate = (correctly_rejected / total_tested) * 100 if total_tested > 0 else 0
print("\n-------------------------------------------------------")
print(f"Sanity Check Summary: {correctly_rejected}/{total_tested} Non-Soil Images Correctly Rejected ({rejection_rate:.1f}%)")
print("-------------------------------------------------------")

if rejection_rate == 100.0:
    print("SUCCESS: Stage 1 Binary Detector reliably rejects 100% of non-soil images!")
else:
    print("WARNING: Some non-soil images passed Stage 1 threshold.")
