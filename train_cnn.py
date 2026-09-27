import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 15

TRAIN_DIR = "dataset/train"
VAL_DIR = "dataset/validation"

# -----------------------------
# LOAD DATA
# -----------------------------

train_data = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=42
)

validation_data = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = train_data.class_names

print("Soil classes:")
print(class_names)

# -----------------------------
# CLASS WEIGHTS
# -----------------------------

class_counts = []

for class_name in class_names:

    folder = f"{TRAIN_DIR}/{class_name}"

    count = len([
        file for file in tf.io.gfile.listdir(folder)
        if file.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    class_counts.append(count)

print("Training image counts:")
print(class_counts)

class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(len(class_names)),
    y=np.concatenate([
        np.full(count, index)
        for index, count in enumerate(class_counts)
    ])
)

class_weights = {
    index: weight
    for index, weight in enumerate(class_weights_array)
}

print("Class weights:")
print(class_weights)

# -----------------------------
# DATA AUGMENTATION
# -----------------------------

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.15),
    layers.RandomContrast(0.15)
])

# -----------------------------
# CNN MODEL
# -----------------------------

base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

base_model.trainable = False

model = models.Sequential([
    layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),

    data_augmentation,

    layers.Lambda(preprocess_input),

    base_model,

    layers.GlobalAveragePooling2D(),

    layers.Dropout(0.4),

    layers.Dense(
        128,
        activation="relu"
    ),

    layers.Dropout(0.3),

    layers.Dense(
        len(class_names),
        activation="softmax"
    )
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# -----------------------------
# TRAIN
# -----------------------------

history = model.fit(
    train_data,
    validation_data=validation_data,
    epochs=EPOCHS,
    class_weight=class_weights
)

# -----------------------------
# SAVE
# -----------------------------

model.save("soil_cnn_4class.keras")

print()
print("Improved CNN training completed!")
print("Model saved as ssoil_cnn_4class.keras")