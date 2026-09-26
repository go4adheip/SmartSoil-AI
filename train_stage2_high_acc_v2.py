import os
import tensorflow as tf
from tensorflow.keras import layers, models, applications
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

print("=================================================================")
print("--- STAGE 2 V2: EFFICIENTNET / MOBILENET HIGH ACCURACY CLASSIFIER ---")
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

# Compute exact class weights
class_counts = [len(os.listdir(os.path.join(train_dir, c))) for c in class_names]
class_weights_arr = compute_class_weight("balanced", classes=np.arange(num_classes), y=np.concatenate([np.full(cnt, idx) for idx, cnt in enumerate(class_counts)]))
class_weights = {i: float(w) for i, w in enumerate(class_weights_arr)}
print("Class weights:", class_weights)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Data Augmentation Layer
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.35),
    layers.RandomZoom(0.25),
    layers.RandomContrast(0.3),
    layers.RandomBrightness(0.2)
])

inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = data_augmentation(inputs)
# EfficientNetB0 expects inputs scaled [0, 255] or handles rescaling internally
base_model = applications.EfficientNetB0(
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

# Phase 1: Train Head with Label Smoothing
print("\n--- PHASE 1: Warmup EfficientNet Classifier Head ---")
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(ignore_class=None),
    metrics=["accuracy"]
)

callbacks_p1 = [
    tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=4, restore_best_weights=True)
]

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    class_weight=class_weights,
    callbacks=callbacks_p1
)

# Phase 2: Unfreeze top 80 layers and fine-tune
print("\n--- PHASE 2: Fine-Tuning EfficientNet Base Layers ---")
base_model.trainable = True
for layer in base_model.layers[:-80]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks_p2 = [
    tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=6, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_accuracy", factor=0.5, patience=2, min_lr=1e-6)
]

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20,
    class_weight=class_weights,
    callbacks=callbacks_p2
)

model.save("soil_cnn.keras")
print("\nHigh accuracy model saved to soil_cnn.keras successfully!")

# Multi-Crop Test Set Evaluation (Ensemble Predict)
print("\n=======================================================")
print("--- HELD-OUT TEST SET EVALUATION (MULTI-CROP ENSEMBLE) ---")
print("=======================================================")

y_true = []
y_pred = []

for images, labels in test_ds:
    # Perform 5-crop evaluation per image in test set
    for i in range(len(images)):
        img_np = images[i].numpy().astype(np.uint8)
        lbl = int(labels[i].numpy())
        
        # 5 crops: Center + 4 corners
        h, w, _ = img_np.shape
        cw, ch = int(w * 0.85), int(h * 0.85)
        crops = [
            img_np[(h-ch)//2:(h+ch)//2, (w-cw)//2:(w+cw)//2],
            img_np[0:ch, 0:cw],
            img_np[0:ch, w-cw:w],
            img_np[h-ch:h, 0:cw],
            img_np[h-ch:h, w-cw:w]
        ]
        
        tensors = []
        for c in crops:
            resized = tf.image.resize(c, (IMG_SIZE, IMG_SIZE)).numpy()
            tensors.append(resized)
            tensors.append(np.fliplr(resized))
            
        tensors_batch = np.array(tensors)
        preds = model.predict(tensors_batch, verbose=0)
        avg_pred = np.mean(preds, axis=0)
        pred_label = int(np.argmax(avg_pred))
        
        y_true.append(lbl)
        y_pred.append(pred_label)

y_true = np.array(y_true)
y_pred = np.array(y_pred)

target_names_readable = [c.replace("_", " ") for c in class_names]
print("\n--- Detailed Classification Report ---")
report_str = classification_report(y_true, y_pred, target_names=target_names_readable, digits=4)
print(report_str)

print("\n--- Confusion Matrix ---")
cm = confusion_matrix(y_true, y_pred)
print("Labels:", target_names_readable)
print(cm)

test_acc = float(np.mean(y_true == y_pred) * 100)
print(f"\nFINAL REAL HELD-OUT TEST ACCURACY: {test_acc:.2f}%")
