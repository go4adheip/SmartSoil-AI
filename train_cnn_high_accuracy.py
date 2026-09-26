import os
import tensorflow as tf
from tensorflow.keras import layers, models, applications
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

IMG_SIZE = 224
BATCH_SIZE = 32
DATASET_DIR = "Soil-Classification-Dataset-main/Orignal-Dataset"

print("Training Single High-Accuracy MobileNetV2 CNN Model...")

# Load training and validation datasets
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
print("Soil Classes found:", class_names)

# Compute balanced class weights
class_counts = []
for cname in class_names:
    folder = os.path.join(DATASET_DIR, cname)
    cnt = len([f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    class_counts.append(cnt)

print("Class Image Counts:", class_counts)

class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(len(class_names)),
    y=np.concatenate([np.full(cnt, idx) for idx, cnt in enumerate(class_counts)])
)
class_weights = {idx: float(w) for idx, w in enumerate(class_weights_array)}
print("Class weights:", class_weights)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Data augmentation
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.2),
])

# Build model with explicit Rescaling layer
inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = data_augmentation(inputs)
x = layers.Rescaling(1./127.5, offset=-1)(x)

base_model = applications.MobileNetV2(
    input_tensor=x,
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False

y = layers.GlobalAveragePooling2D()(base_model.output)
y = layers.BatchNormalization()(y)
y = layers.Dropout(0.3)(y)
y = layers.Dense(128, activation="relu")(y)
y = layers.Dropout(0.2)(y)
outputs = layers.Dense(len(class_names), activation="softmax")(y)

model = models.Model(inputs, outputs)

# Phase 1: Train Head
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\n--- PHASE 1: Training Classification Head (8 Epochs) ---")
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=8,
    class_weight=class_weights
)

# Phase 2: Fine-Tuning top MobileNetV2 layers
print("\n--- PHASE 2: Fine-Tuning MobileNetV2 Base Layers (8 Epochs) ---")
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=8,
    class_weight=class_weights
)

# Save high-accuracy model
model.save("soil_cnn.keras")
print("\nHigh-Accuracy Soil CNN Model trained & saved to soil_cnn.keras!")
