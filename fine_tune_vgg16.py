import os
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ============================================================
# 1. SETTINGS
# ============================================================

TRAIN_DIR = "dataset/train"
VALIDATION_DIR = "dataset/validation"
TEST_DIR = "dataset/test"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

OLD_MODEL_PATH = "models/vgg16_pneumonia.keras"
NEW_MODEL_PATH = "models/vgg16_finetuned.keras"


# ============================================================
# 2. LOAD DATA
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
# 3. PRINT CLASS MAPPING
# ============================================================

print("\nClass mapping:")
print(train_data.class_indices)


# ============================================================
# 4. LOAD VGG16 BASE MODEL
# ============================================================

base_model = VGG16(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)


# ============================================================
# 5. FREEZE EARLY LAYERS
# ============================================================

base_model.trainable = True

# Freeze the first 15 layers
for layer in base_model.layers[:15]:
    layer.trainable = False


# ============================================================
# 6. SHOW TRAINABLE LAYERS
# ============================================================

print("\nVGG16 layer status:")
print("===================")

for layer in base_model.layers:
    print(
        layer.name,
        "->",
        "Trainable" if layer.trainable else "Frozen"
    )


# ============================================================
# 7. CREATE CLASSIFICATION HEAD
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


model = Model(
    inputs=base_model.input,
    outputs=output
)


# ============================================================
# 8. COMPILE WITH SMALL LEARNING RATE
# ============================================================

model.compile(
    optimizer=Adam(
        learning_rate=0.00001
    ),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


# ============================================================
# 9. MODEL SUMMARY
# ============================================================

model.summary()


# ============================================================
# 10. CALLBACKS
# ============================================================

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=2,
    restore_best_weights=True
)


checkpoint = ModelCheckpoint(
    NEW_MODEL_PATH,
    monitor="val_loss",
    save_best_only=True
)


# ============================================================
# 11. TRAIN / FINE-TUNE
# ============================================================

EPOCHS = 10

history = model.fit(
    train_data,
    validation_data=validation_data,
    epochs=EPOCHS,
    callbacks=[
        early_stop,
        checkpoint
    ]
)


# ============================================================
# 12. LOAD BEST MODEL
# ============================================================

model = load_model(
    NEW_MODEL_PATH
)


# ============================================================
# 13. EVALUATE ON TEST DATA
# ============================================================

test_loss, test_accuracy = model.evaluate(
    test_data,
    verbose=1
)


# ============================================================
# 14. PREDICTIONS
# ============================================================

test_data.reset()

predictions = model.predict(
    test_data,
    verbose=1
)

y_pred = (
    predictions > 0.5
).astype(int).flatten()

y_true = test_data.classes


# ============================================================
# 15. CALCULATE METRICS
# ============================================================

accuracy = accuracy_score(
    y_true,
    y_pred
)

precision = precision_score(
    y_true,
    y_pred,
    pos_label=1,
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    pos_label=1,
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    pos_label=1,
    zero_division=0
)


# ============================================================
# 16. PRINT RESULTS
# ============================================================

print("\n")
print("=" * 55)
print("VGG16 FINE-TUNING TEST RESULTS")
print("=" * 55)

print(f"Test Loss      : {test_loss:.4f}")
print(f"Test Accuracy  : {accuracy * 100:.2f}%")
print(f"Precision      : {precision * 100:.2f}%")
print(f"Recall         : {recall * 100:.2f}%")
print(f"F1-score       : {f1 * 100:.2f}%")


# ============================================================
# 17. CLASSIFICATION REPORT
# ============================================================

class_names = list(
    test_data.class_indices.keys()
)

print("\n")
print("=" * 55)
print("CLASSIFICATION REPORT")
print("=" * 55)

print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0
    )
)


# ============================================================
# 18. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred
)

print("\n")
print("=" * 55)
print("CONFUSION MATRIX")
print("=" * 55)

print(cm)


# ============================================================
# 19. SAVE CONFUSION MATRIX
# ============================================================

os.makedirs(
    "results",
    exist_ok=True
)

plt.figure(figsize=(7, 6))

plt.imshow(
    cm,
    interpolation="nearest"
)

plt.title(
    "VGG16 Fine-Tuned Confusion Matrix"
)

plt.colorbar()

plt.xticks(
    range(len(class_names)),
    class_names
)

plt.yticks(
    range(len(class_names)),
    class_names
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")


for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )


plt.tight_layout()

plt.savefig(
    "results/vgg16_finetuned_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 20. SAVE TRAINING GRAPHS
# ============================================================

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

plt.title(
    "VGG16 Fine-Tuning Accuracy"
)

plt.legend()

plt.savefig(
    "results/vgg16_finetuned_accuracy.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


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

plt.title(
    "VGG16 Fine-Tuning Loss"
)

plt.legend()

plt.savefig(
    "results/vgg16_finetuned_loss.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 21. FINAL MESSAGE
# ============================================================

print("\n")
print("=" * 55)
print("FINE-TUNING COMPLETED")
print("=" * 55)

print(
    "Best model saved at:"
)

print(
    NEW_MODEL_PATH
)

print(
    "\nResults saved in the 'results' folder."
)