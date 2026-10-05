import os
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
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
# 1. PATHS
# ============================================================

BASE_DIR = r"C:\project\Pneumonia_Detection_Project"

TRAIN_DIR = os.path.join(BASE_DIR, "dataset_final", "train")
VAL_DIR = os.path.join(BASE_DIR, "dataset_final", "validation")
TEST_DIR = os.path.join(BASE_DIR, "dataset_final", "test")

MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "vgg16_finetuned_final.keras"
)


# ============================================================
# 2. SETTINGS
# ============================================================

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15

# Classification:
# NORMAL = 0
# PNEUMONIA = 1


# ============================================================
# 3. DATA AUGMENTATION
# ============================================================

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,

    rotation_range=10,
    zoom_range=0.10,
    width_shift_range=0.05,
    height_shift_range=0.05,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)


# ============================================================
# 4. LOAD DATA
# ============================================================

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=True,
    seed=42
)

validation_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)


print("\n==========================================")
print("CLASS INDICES")
print("==========================================")
print(train_generator.class_indices)


print("\n==========================================")
print("DATASET SIZES")
print("==========================================")
print("Training images   :", train_generator.samples)
print("Validation images :", validation_generator.samples)
print("Test images       :", test_generator.samples)


# ============================================================
# 5. LOAD VGG16 BASE MODEL
# ============================================================

print("\nLoading VGG16...")

base_model = VGG16(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)


# ============================================================
# 6. FREEZE / FINE-TUNE VGG16
# ============================================================

# Freeze the earlier layers.
# Fine-tune the later convolutional layers.

for layer in base_model.layers:
    layer.trainable = False


# Unfreeze the last 4 convolutional layers
for layer in base_model.layers[-4:]:
    layer.trainable = True


print("\n==========================================")
print("VGG16 LAYER STATUS")
print("==========================================")

for layer in base_model.layers:
    print(
        f"{layer.name:25s} "
        f"Trainable: {layer.trainable}"
    )


# ============================================================
# 7. BUILD CLASSIFICATION HEAD
# ============================================================

x = base_model.output

x = GlobalAveragePooling2D()(x)

x = Dense(
    256,
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
# 8. COMPILE MODEL
# ============================================================

model.compile(
    optimizer=Adam(
        learning_rate=1e-5
    ),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


print("\n==========================================")
print("MODEL SUMMARY")
print("==========================================")

model.summary()


# ============================================================
# 9. CALLBACKS
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
# 10. TRAIN MODEL
# ============================================================

print("\n==========================================")
print("STARTING VGG16 FINE-TUNING")
print("==========================================")

history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=EPOCHS,
    callbacks=[
        early_stopping,
        model_checkpoint
    ]
)


# ============================================================
# 11. LOAD BEST MODEL
# ============================================================

print("\nLoading best saved model...")

from tensorflow.keras.models import load_model

model = load_model(MODEL_PATH)

print("Best model loaded successfully.")


# ============================================================
# 12. TEST EVALUATION
# ============================================================

print("\n==========================================")
print("TEST EVALUATION")
print("==========================================")

test_loss, test_accuracy = model.evaluate(
    test_generator,
    verbose=1
)

print("\nTest Loss     :", round(test_loss, 4))
print(
    "Test Accuracy :",
    round(test_accuracy * 100, 2),
    "%"
)


# ============================================================
# 13. PREDICTIONS
# ============================================================

print("\nGenerating predictions...")

test_generator.reset()

probabilities = model.predict(
    test_generator,
    verbose=1
).reshape(-1)

# Default threshold
threshold = 0.50

predictions = (
    probabilities >= threshold
).astype(int)

true_labels = test_generator.classes


# ============================================================
# 14. CLASSIFICATION METRICS
# ============================================================

accuracy = accuracy_score(
    true_labels,
    predictions
)

precision = precision_score(
    true_labels,
    predictions,
    zero_division=0
)

pneumonia_recall = recall_score(
    true_labels,
    predictions,
    zero_division=0
)

f1 = f1_score(
    true_labels,
    predictions,
    zero_division=0
)


# Calculate NORMAL recall / specificity

cm = confusion_matrix(
    true_labels,
    predictions,
    labels=[0, 1]
)

tn, fp, fn, tp = cm.ravel()

normal_recall = tn / (tn + fp)


# ============================================================
# 15. PRINT FINAL RESULTS
# ============================================================

print("\n==========================================")
print("FINAL TEST RESULTS")
print("==========================================")

print(
    f"Accuracy              : {accuracy * 100:.2f}%"
)

print(
    f"Precision             : {precision * 100:.2f}%"
)

print(
    f"PNEUMONIA Recall      : "
    f"{pneumonia_recall * 100:.2f}%"
)

print(
    f"NORMAL Recall         : "
    f"{normal_recall * 100:.2f}%"
)

print(
    f"F1 Score              : "
    f"{f1 * 100:.2f}%"
)


# ============================================================
# 16. CONFUSION MATRIX
# ============================================================

print("\n==========================================")
print("CONFUSION MATRIX")
print("==========================================")

print(
    "                 Predicted NORMAL   "
    "Predicted PNEUMONIA"
)

print(
    f"Actual NORMAL       {tn:5d}              {fp:5d}"
)

print(
    f"Actual PNEUMONIA    {fn:5d}              {tp:5d}"
)


# ============================================================
# 17. CLASSIFICATION REPORT
# ============================================================

print("\n==========================================")
print("CLASSIFICATION REPORT")
print("==========================================")

print(
    classification_report(
        true_labels,
        predictions,
        target_names=[
            "NORMAL",
            "PNEUMONIA"
        ],
        digits=4,
        zero_division=0
    )
)


# ============================================================
# 18. SAVE RESULTS TO TXT
# ============================================================

results_file = os.path.join(
    RESULTS_DIR,
    "vgg16_finetuned_final_results.txt"
)

with open(results_file, "w") as f:

    f.write(
        "VGG16 FINE-TUNED FINAL RESULTS\n"
    )

    f.write(
        "====================================\n\n"
    )

    f.write(
        f"Model: vgg16_finetuned_final.keras\n"
    )

    f.write(
        f"Threshold: {threshold}\n\n"
    )

    f.write(
        f"Test Loss: {test_loss:.4f}\n"
    )

    f.write(
        f"Accuracy: {accuracy * 100:.2f}%\n"
    )

    f.write(
        f"Precision: {precision * 100:.2f}%\n"
    )

    f.write(
        f"PNEUMONIA Recall: "
        f"{pneumonia_recall * 100:.2f}%\n"
    )

    f.write(
        f"NORMAL Recall: "
        f"{normal_recall * 100:.2f}%\n"
    )

    f.write(
        f"F1 Score: {f1 * 100:.2f}%\n\n"
    )

    f.write("CONFUSION MATRIX\n")
    f.write("----------------\n")

    f.write(
        f"TN = {tn}\n"
        f"FP = {fp}\n"
        f"FN = {fn}\n"
        f"TP = {tp}\n\n"
    )

    f.write("CLASSIFICATION REPORT\n")
    f.write("---------------------\n")

    f.write(
        classification_report(
            true_labels,
            predictions,
            target_names=[
                "NORMAL",
                "PNEUMONIA"
            ],
            digits=4,
            zero_division=0
        )
    )


print(
    f"\nResults saved to:\n{results_file}"
)


# ============================================================
# 19. PLOT TRAINING / VALIDATION ACCURACY
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

plt.title(
    "VGG16 Fine-Tuning Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.legend()

plt.grid(True)

accuracy_plot = os.path.join(
    RESULTS_DIR,
    "vgg16_finetuned_final_accuracy.png"
)

plt.savefig(
    accuracy_plot,
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 20. PLOT TRAINING / VALIDATION LOSS
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

plt.title(
    "VGG16 Fine-Tuning Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend()

plt.grid(True)

loss_plot = os.path.join(
    RESULTS_DIR,
    "vgg16_finetuned_final_loss.png"
)

plt.savefig(
    loss_plot,
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 21. FINAL MESSAGE
# ============================================================

print("\n==========================================")
print("TRAINING AND EVALUATION COMPLETED")
print("==========================================")

print(
    "\nSaved model:"
)

print(MODEL_PATH)

print(
    "\nSaved results:"
)

print(results_file)

print(
    "\nSaved plots:"
)

print(accuracy_plot)
print(loss_plot)