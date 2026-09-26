import os
import tensorflow as tf
from tensorflow.keras import layers, models, applications
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

print("=================================================================")
print("--- STAGE 2: HIGH ACCURACY MULTICLASSIFIER TRAINING & TEST EVAL ---")
print("=================================================================")

IMG_SIZE = 224
BATCH_SIZE = 32

base_split_dir = "dataset/multiclass_split"
train_dir = os.path.join(base_split_dir, "train")
val_dir = os.path.join(base_split_dir, "val")
test_dir = os.path.join(base_split_dir, "test")

train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=42
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = train_ds.class_names
num_classes = len(class_names)
print("\nSoil Classes:", class_names)

# Class weight calculation based on raw training distribution before augmentation
class_counts = [len(os.listdir(os.path.join(train_dir, c))) for c in class_names]
class_weights_arr = compute_class_weight("balanced", classes=np.arange(num_classes), y=np.concatenate([np.full(cnt, idx) for idx, cnt in enumerate(class_counts)]))
class_weights = {i: float(w) for i, w in enumerate(class_weights_arr)}

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Data augmentation pipeline
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.3),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.25),
    layers.RandomTranslation(height_factor=0.1, width_factor=0.1)
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
y = layers.Dropout(0.4)(y)
y = layers.Dense(256, activation="relu")(y)
y = layers.BatchNormalization()(y)
y = layers.Dropout(0.3)(y)
outputs = layers.Dense(num_classes, activation="softmax")(y)

model = models.Model(inputs, outputs)

# Phase 1: Train Head
print("\n--- PHASE 1: Training Classification Head ---")
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks_p1 = [
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)
]

history_p1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    class_weight=class_weights,
    callbacks=callbacks_p1
)

# Phase 2: Fine-Tune Top 60 Layers
print("\n--- PHASE 2: Deep Fine-Tuning Top 60 Layers of MobileNetV2 ---")
base_model.trainable = True
for layer in base_model.layers[:-60]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks_p2 = [
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6)
]

history_p2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15,
    class_weight=class_weights,
    callbacks=callbacks_p2
)

model.save("soil_cnn.keras")
print("\nModel saved to soil_cnn.keras successfully!")

# Evaluation on Genuinely Held-Out Test Set
print("\n=======================================================")
print("--- HELD-OUT TEST SET EVALUATION RESULTS ---")
print("=======================================================")

y_true = []
y_pred = []

for images, labels in test_ds:
    preds = model.predict(images, verbose=0)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(preds, axis=1))

y_true = np.array(y_true)
y_pred = np.array(y_pred)

print("\n--- Detailed Classification Report ---")
target_names_readable = [c.replace("_", " ") for c in class_names]
report_str = classification_report(y_true, y_pred, target_names=target_names_readable, digits=4)
print(report_str)

print("\n--- Confusion Matrix ---")
cm = confusion_matrix(y_true, y_pred)
print("Labels:", target_names_readable)
print(cm)

test_acc = float(np.mean(y_true == y_pred) * 100)
print(f"\nFINAL REAL HELD-OUT TEST ACCURACY: {test_acc:.2f}%")
