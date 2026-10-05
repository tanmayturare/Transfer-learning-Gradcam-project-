import tensorflow as tf
import os
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D


# ============================================================
# 1. DATASET PATHS
# ============================================================

TRAIN_DIR = "dataset/train"
VALIDATION_DIR = "dataset/validation"
TEST_DIR = "dataset/test"


# ============================================================
# 2. IMAGE SETTINGS
# ============================================================

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32


# ============================================================
# 3. DATA PREPROCESSING
# ============================================================

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=10,
    zoom_range=0.1,
    horizontal_flip=True
)

validation_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0
)

test_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0
)


# ============================================================
# 4. LOAD DATASET
# ============================================================

train_data = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

validation_data = validation_datagen.flow_from_directory(
    VALIDATION_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

test_data = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)


# ============================================================
# 5. DISPLAY CLASS MAPPING
# ============================================================

print("\nClass mapping:")
print(train_data.class_indices)


# ============================================================
# 6. LOAD PRETRAINED VGG16
# ============================================================

base_model = VGG16(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)


# ============================================================
# 7. FREEZE VGG16 LAYERS
# ============================================================

base_model.trainable = False


# ============================================================
# 8. ADD OUR CLASSIFIER
# ============================================================

x = base_model.output

x = GlobalAveragePooling2D()(x)

x = Dense(
    128,
    activation="relu"
)(x)

x = Dropout(0.5)(x)

output = Dense(
    1,
    activation="sigmoid"
)(x)


# ============================================================
# 9. CREATE FINAL MODEL
# ============================================================

model = Model(
    inputs=base_model.input,
    outputs=output
)


# ============================================================
# 10. COMPILE MODEL
# ============================================================

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


# ============================================================
# 11. DISPLAY MODEL
# ============================================================

model.summary()


# ============================================================
# 12. TRAIN MODEL
# ============================================================

EPOCHS = 5

history = model.fit(
    train_data,
    validation_data=validation_data,
    epochs=EPOCHS
)


# ============================================================
# 13. SAVE MODEL
# ============================================================

os.makedirs("models", exist_ok=True)

model.save(
    "models/vgg16_pneumonia.keras"
)


# ============================================================
# 14. PLOT ACCURACY
# ============================================================

os.makedirs("results", exist_ok=True)

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("VGG16 Transfer Learning Accuracy")
plt.legend()

plt.savefig(
    "results/vgg16_accuracy.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 15. PLOT LOSS
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("VGG16 Transfer Learning Loss")
plt.legend()

plt.savefig(
    "results/vgg16_loss.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()