import os
import tensorflow as tf
from tensorflow.keras import layers, models, applications

print("--- STAGE 1: Training Binary Soil vs Non-Soil Detector ---")

IMG_SIZE = 224
BATCH_SIZE = 32
DATASET_DIR = "dataset/binary_dataset"

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
print("Binary Classes found:", class_names)
# Class 0: non_soil, Class 1: soil

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.2),
])

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
y = layers.Dense(64, activation="relu")(y)
outputs = layers.Dense(1, activation="sigmoid")(y)

model = models.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
]

print("\n--- PHASE 1: Training Binary Classification Head ---")
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=6,
    callbacks=callbacks
)

print("\n--- PHASE 2: Fine-Tuning MobileNetV2 Base ---")
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=6,
    callbacks=callbacks
)

model.save("soil_binary_detector.keras")
print("\nSTAGE 1 Binary Model trained & saved to soil_binary_detector.keras!")
