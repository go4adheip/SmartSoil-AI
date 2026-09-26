import os
import shutil
import random

SOURCE = "Soil-Classification-Dataset-main/Orignal-Dataset"
DESTINATION = "dataset"

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

random.seed(42)

classes = [
    "Alluvial_Soil",
    "Arid_Soil",
    "Black_Soil",
    "Laterite_Soil",
    "Mountain_Soil",
    "Red_Soil",
    "Yellow_Soil"
]

for soil_class in classes:

    source_folder = os.path.join(SOURCE, soil_class)

    train_folder = os.path.join(DESTINATION, "train", soil_class)
    validation_folder = os.path.join(DESTINATION, "validation", soil_class)
    test_folder = os.path.join(DESTINATION, "test", soil_class)

    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(validation_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    images = [
        file for file in os.listdir(source_folder)
        if file.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    random.shuffle(images)

    total = len(images)

    train_end = int(total * TRAIN_RATIO)
    validation_end = train_end + int(total * VALIDATION_RATIO)

    train_images = images[:train_end]
    validation_images = images[train_end:validation_end]
    test_images = images[validation_end:]

    for image in train_images:
        shutil.copy2(
            os.path.join(source_folder, image),
            os.path.join(train_folder, image)
        )

    for image in validation_images:
        shutil.copy2(
            os.path.join(source_folder, image),
            os.path.join(validation_folder, image)
        )

    for image in test_images:
        shutil.copy2(
            os.path.join(source_folder, image),
            os.path.join(test_folder, image)
        )

    print(
        soil_class,
        "->",
        len(train_images),
        "train,",
        len(validation_images),
        "validation,",
        len(test_images),
        "test"
    )

print()
print("Dataset splitting completed!")