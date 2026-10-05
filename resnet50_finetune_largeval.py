
import os
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# 1. SETTINGS
# ============================================================

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10

TRAIN_DIR = "dataset_final/train"
VALIDATION_DIR = "dataset_final/validation"
TEST_DIR = "dataset_final/test"

MODEL_DIR = "models"
RESULTS_DIR = "results"

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "resnet50_finetuned_largeval.keras"
)

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# 2. DATA GENERATORS
# ============================================================

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=10,
    zoom_range=0.1,
    horizontal_flip=True
)

validation_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)


train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=True,
    seed=42
)

validation_generator = validation_datagen.flow_from_directory(
    VALIDATION_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)


print("\nClass mapping:", train_generator.class_indices)

assert train_generator.class_indices == validation_generator.class_indices
assert train_generator.class_indices == test_generator.class_indices

print("Training images:", train_generator.samples)
print("Validation images:", validation_generator.samples)
print("Test images:", test_generator.samples)


# ============================================================
# 3. LOAD RESNET50
# ============================================================

base_model = ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

# Enable fine-tuning
base_model.trainable = True

# Freeze the first 140 layers
for layer in base_model.layers[:140]:
    layer.trainable = False


# ============================================================
# 4. BUILD CLASSIFICATION HEAD
# ============================================================

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.5)(x)
output = Dense(1, activation="sigmoid")(x)

model = Model(
    inputs=base_model.input,
    outputs=output
)


# ============================================================
# 5. COMPILE MODEL
# ============================================================

model.compile(
    optimizer=Adam(learning_rate=0.00001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

print("\nTrainable parameters and model summary:")
model.summary()


# ============================================================
# 6. CALLBACKS
# ============================================================

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True,
    verbose=1
)

checkpoint = ModelCheckpoint(
    filepath=MODEL_PATH,
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)


# ============================================================
# 7. TRAIN MODEL
# ============================================================

print("\n" + "=" * 60)
print("RESNET50 FINE-TUNING - LARGE VALIDATION SET")
print("=" * 60)

history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=EPOCHS,
    callbacks=[early_stop, checkpoint]
)


# ============================================================
# 8. LOAD BEST CHECKPOINT
# ============================================================

print("\nLoading best checkpoint...")
model = load_model(MODEL_PATH)


# ============================================================
# 9. PLOT ACCURACY
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

plt.title("ResNet50 Fine-Tuning - Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "resnet50_finetuned_largeval_accuracy.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 10. PLOT LOSS
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

plt.title("ResNet50 Fine-Tuning - Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "resnet50_finetuned_largeval_loss.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 11. EVALUATE ON TEST SET
# ============================================================

print("\n" + "=" * 60)
print("FINAL TEST EVALUATION")
print("=" * 60)

test_generator.reset()

test_loss, keras_accuracy = model.evaluate(
    test_generator,
    verbose=1
)

test_generator.reset()

probabilities = model.predict(
    test_generator,
    verbose=1
).ravel()

predictions = (probabilities >= 0.5).astype(int)
true_labels = test_generator.classes


# ============================================================
# 12. CALCULATE METRICS
# ============================================================

accuracy = accuracy_score(true_labels, predictions)

precision = precision_score(
    true_labels, predictions, zero_division=0
)

recall = recall_score(
    true_labels, predictions, zero_division=0
)

f1 = f1_score(
    true_labels, predictions, zero_division=0
)


# ============================================================
# 13. PRINT RESULTS
# ============================================================

print("\n" + "=" * 60)
print("RESNET50 LARGE-VALIDATION TEST RESULTS")
print("=" * 60)

print(f"Test Loss      : {test_loss:.4f}")
print(f"Test Accuracy  : {accuracy * 100:.2f}%")
print(f"Precision      : {precision * 100:.2f}%")
print(f"Recall         : {recall * 100:.2f}%")
print(f"F1-score       : {f1 * 100:.2f}%")


print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        true_labels,
        predictions,
        labels=[0, 1],
        target_names=["NORMAL", "PNEUMONIA"],
        zero_division=0
    )
)


# ============================================================
# 14. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    true_labels,
    predictions,
    labels=[0, 1]
)

print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print(cm)

plt.figure(figsize=(6, 5))
plt.imshow(cm)
plt.title("ResNet50 Large Validation - Confusion Matrix")
plt.colorbar()

plt.xticks([0, 1], ["NORMAL", "PNEUMONIA"])
plt.yticks([0, 1], ["NORMAL", "PNEUMONIA"])

plt.xlabel("Predicted Label")
plt.ylabel("True Label")

for i in range(2):
    for j in range(2):
        plt.text(
            j, i, str(cm[i, j]),
            ha="center",
            va="center"
        )

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "resnet50_finetuned_largeval_confusion_matrix.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 15. COMPLETION
# ============================================================

print("\n" + "=" * 60)
print("TRAINING AND EVALUATION COMPLETED")
print("=" * 60)

print("Saved model:", MODEL_PATH)
print("Saved accuracy graph.")
print("Saved loss graph.")
print("Saved confusion matrix.")