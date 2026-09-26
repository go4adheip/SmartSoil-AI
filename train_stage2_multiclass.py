import os
import tensorflow as tf
from tensorflow.keras import layers, models, applications
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

print("--- STAGE 2: Training 7-Class Soil Classifier to Convergence ---")

IMG_SIZE = 224
BATCH_SIZE = 32
DATASET_DIR = "Soil-Classification-Dataset-main/Orignal-Dataset"

# 1. Load full dataset as raw dataset
full_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    seed=42,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=True
)

class_names = full_ds.class_names
num_classes = len(class_names)
print("Soil Classes found:", class_names)

# 2. Partition into 70% Train, 15% Val, 15% Test
total_batches = len(full_ds)
train_size = int(0.70 * total_batches)
val_size = int(0.15 * total_batches)
test_size = total_batches - train_size - val_size

train_ds = full_ds.take(train_size)
val_ds = full_ds.skip(train_size).take(val_size)
test_ds = full_ds.skip(train_size + val_size)

print(f"Dataset split: Train = {train_size} batches, Val = {val_size} batches, Test = {test_size} batches")

# Calculate class weights for balance
class_counts = [len(os.listdir(os.path.join(DATASET_DIR, c))) for c in class_names]
class_weights_arr = compute_class_weight("balanced", classes=np.arange(num_classes), y=np.concatenate([np.full(cnt, idx) for idx, cnt in enumerate(class_counts)]))
class_weights = {i: float(w) for i, w in enumerate(class_weights_arr)}

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.25),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.25),
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
y = layers.Dropout(0.35)(y)
y = layers.Dense(128, activation="relu")(y)
y = layers.Dropout(0.25)(y)
outputs = layers.Dense(num_classes, activation="softmax")(y)

model = models.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)
]

print("\n--- Phase 1: Training Classification Head ---")
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    class_weight=class_weights,
    callbacks=callbacks
)

print("\n--- Phase 2: Fine-Tuning Top MobileNetV2 Layers ---")
base_model.trainable = True
for layer in base_model.layers[:-40]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    class_weight=class_weights,
    callbacks=callbacks
)

# 3. Evaluate on Held-Out Test Set
print("\n=======================================================")
print("--- HELD-OUT TEST SET EVALUATION METRICS ---")
print("=======================================================")

y_true = []
y_pred = []

for images, labels in test_ds:
    preds = model.predict(images, verbose=0)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(preds, axis=1))

print("\n--- Classification Report (Precision, Recall, F1-Score) ---")
print(classification_report(y_true, y_pred, target_names=class_names))

print("\n--- Confusion Matrix ---")
cm = confusion_matrix(y_true, y_pred)
print(cm)

model.save("soil_cnn.keras")
print("\nSTAGE 2 Multiclass Model trained & saved to soil_cnn.keras!")
