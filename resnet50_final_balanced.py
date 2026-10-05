import os
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ============================================================
# SETTINGS
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
    "resnet50_final_balanced.keras"
)

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# 1. DATA GENERATORS
# ============================================================

print("\n" + "=" * 60)
print("LOADING DATASET")
print("=" * 60)

# Training data augmentation
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=10,
    zoom_range=0.1,
    horizontal_flip=True
)

# Validation data - no augmentation
validation_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

# Test data - no augmentation
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


print("\nClass Mapping:")
print(train_generator.class_indices)

print("\nDataset Summary:")
print("Training images   :", train_generator.samples)
print("Validation images :", validation_generator.samples)
print("Test images       :", test_generator.samples)


# ============================================================
# 2. CALCULATE CLASS WEIGHTS
# ============================================================

print("\n" + "=" * 60)
print("CALCULATING CLASS WEIGHTS")
print("=" * 60)

# Class indices:
# NORMAL = 0
# PNEUMONIA = 1

classes = np.array([0, 1])

class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=train_generator.classes
)

class_weights = {
    0: class_weights_array[0],
    1: class_weights_array[1]
}

print("\nClass Weights:")
print("NORMAL     :", class_weights[0])
print("PNEUMONIA  :", class_weights[1])

print("\nInterpretation:")
print("A higher weight means mistakes for that class")
print("are given more importance during training.")


# ============================================================
# 3. BUILD RESNET50 MODEL
# ============================================================

print("\n" + "=" * 60)
print("BUILDING RESNET50 MODEL")
print("=" * 60)

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


# Classification head
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
# 4. COMPILE MODEL
# ============================================================

model.compile(
    optimizer=Adam(learning_rate=0.00001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


print("\nModel Summary:")
model.summary()


# ============================================================
# 5. CALLBACKS
# ============================================================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True,
    verbose=1
)

model_checkpoint = ModelCheckpoint(
    MODEL_PATH,
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)


# ============================================================
# 6. TRAIN MODEL
# ============================================================

print("\n" + "=" * 60)
print("STARTING BALANCED RESNET50 TRAINING")
print("=" * 60)

print("\nTraining configuration:")
print("Image size        :", IMAGE_SIZE)
print("Batch size        :", BATCH_SIZE)
print("Maximum epochs    :", EPOCHS)
print("Learning rate     :", 0.00001)
print("Frozen layers     :", 140)
print("Class weights     :", class_weights)


history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=[
        early_stopping,
        model_checkpoint
    ]
)


# ============================================================
# 7. LOAD BEST MODEL
# ============================================================

print("\n" + "=" * 60)
print("LOADING BEST MODEL")
print("=" * 60)

from tensorflow.keras.models import load_model

best_model = load_model(MODEL_PATH)

print("\nBest model loaded from:")
print(MODEL_PATH)


# ============================================================
# 8. EVALUATE ON TEST DATA
# ============================================================

print("\n" + "=" * 60)
print("EVALUATING ON TEST DATA")
print("=" * 60)

test_loss, test_accuracy = best_model.evaluate(
    test_generator,
    verbose=1
)


# ============================================================
# 9. PREDICTIONS
# ============================================================

print("\nGenerating predictions...")

test_generator.reset()

predictions = best_model.predict(
    test_generator,
    verbose=1
)

# Convert probabilities to binary predictions
y_pred = (predictions >= 0.5).astype(int).flatten()

# Actual labels
y_true = test_generator.classes


# ============================================================
# 10. CALCULATE METRICS
# ============================================================

accuracy = accuracy_score(
    y_true,
    y_pred
)

precision = precision_score(
    y_true,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    zero_division=0
)


# ============================================================
# 11. PRINT FINAL RESULTS
# ============================================================

print("\n" + "=" * 60)
print("FINAL TEST RESULTS")
print("=" * 60)

print(f"\nTest Loss      : {test_loss:.4f}")
print(f"Test Accuracy  : {accuracy * 100:.2f}%")
print(f"Precision      : {precision * 100:.2f}%")
print(f"Recall         : {recall * 100:.2f}%")
print(f"F1-score       : {f1 * 100:.2f}%")


# ============================================================
# 12. CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

report = classification_report(
    y_true,
    y_pred,
    target_names=["NORMAL", "PNEUMONIA"],
    zero_division=0
)

print(report)


# ============================================================
# 13. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred
)

print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print(cm)


# ============================================================
# 14. SAVE CONFUSION MATRIX GRAPH
# ============================================================

plt.figure(figsize=(7, 6))

plt.imshow(
    cm,
    interpolation="nearest"
)

plt.title("ResNet50 Balanced - Confusion Matrix")

plt.colorbar()

class_names = ["NORMAL", "PNEUMONIA"]

plt.xticks(
    [0, 1],
    class_names
)

plt.yticks(
    [0, 1],
    class_names
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")


# Add numbers inside matrix
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

cm_path = os.path.join(
    RESULTS_DIR,
    "resnet50_final_balanced_confusion_matrix.png"
)

plt.savefig(
    cm_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nConfusion matrix saved:")
print(cm_path)


# ============================================================
# 15. TRAINING ACCURACY GRAPH
# ============================================================

plt.figure(figsize=(8, 6))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("ResNet50 Balanced - Training vs Validation Accuracy")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.legend()

plt.grid(True)

plt.tight_layout()

accuracy_graph_path = os.path.join(
    RESULTS_DIR,
    "resnet50_final_balanced_accuracy.png"
)

plt.savefig(
    accuracy_graph_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Accuracy graph saved:")
print(accuracy_graph_path)


# ============================================================
# 16. TRAINING LOSS GRAPH
# ============================================================

plt.figure(figsize=(8, 6))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("ResNet50 Balanced - Training vs Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend()

plt.grid(True)

plt.tight_layout()

loss_graph_path = os.path.join(
    RESULTS_DIR,
    "resnet50_final_balanced_loss.png"
)

plt.savefig(
    loss_graph_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Loss graph saved:")
print(loss_graph_path)


# ============================================================
# 17. SAVE RESULTS TO TEXT FILE
# ============================================================

results_path = os.path.join(
    RESULTS_DIR,
    "resnet50_final_balanced_results.txt"
)

with open(results_path, "w") as f:

    f.write("=" * 60 + "\n")
    f.write("RESNET50 BALANCED FINAL MODEL RESULTS\n")
    f.write("=" * 60 + "\n\n")

    f.write(f"Test Loss      : {test_loss:.4f}\n")
    f.write(f"Test Accuracy  : {accuracy * 100:.2f}%\n")
    f.write(f"Precision      : {precision * 100:.2f}%\n")
    f.write(f"Recall         : {recall * 100:.2f}%\n")
    f.write(f"F1-score       : {f1 * 100:.2f}%\n\n")

    f.write("=" * 60 + "\n")
    f.write("CLASSIFICATION REPORT\n")
    f.write("=" * 60 + "\n\n")

    f.write(report)

    f.write("\n\n")
    f.write("=" * 60 + "\n")
    f.write("CONFUSION MATRIX\n")
    f.write("=" * 60 + "\n\n")

    f.write(str(cm))

    f.write("\n\n")
    f.write("=" * 60 + "\n")
    f.write("CLASS WEIGHTS\n")
    f.write("=" * 60 + "\n\n")

    f.write(f"NORMAL     : {class_weights[0]:.4f}\n")
    f.write(f"PNEUMONIA  : {class_weights[1]:.4f}\n")


print("\nResults saved:")
print(results_path)


# ============================================================
# 18. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("TRAINING AND EVALUATION COMPLETED")
print("=" * 60)

print("\nBest model:")
print(MODEL_PATH)

print("\nFinal Test Performance:")
print(f"Accuracy  : {accuracy * 100:.2f}%")
print(f"Precision : {precision * 100:.2f}%")
print(f"Recall    : {recall * 100:.2f}%")
print(f"F1-score  : {f1 * 100:.2f}%")

print("\nGenerated files:")
print("- " + MODEL_PATH)
print("- " + cm_path)
print("- " + accuracy_graph_path)
print("- " + loss_graph_path)
print("- " + results_path)

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)