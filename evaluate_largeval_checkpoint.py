import os

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

TEST_DIR = "dataset_final/test"

MODEL_PATH = "models/resnet50_finetuned_largeval.keras"


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading saved ResNet50 model...")

model = load_model(MODEL_PATH)

print("Model loaded successfully.")


# ============================================================
# TEST DATA
# ============================================================

test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

print("\nClass mapping:")
print(test_generator.class_indices)

print("\nTest images:", test_generator.samples)


# ============================================================
# EVALUATION
# ============================================================

print("\n" + "=" * 60)
print("SAVED LARGE-VALIDATION MODEL EVALUATION")
print("=" * 60)

test_generator.reset()

test_loss, keras_accuracy = model.evaluate(
    test_generator,
    verbose=1
)


# ============================================================
# PREDICTIONS
# ============================================================

test_generator.reset()

probabilities = model.predict(
    test_generator,
    verbose=1
).ravel()

predictions = (
    probabilities >= 0.5
).astype(int)

true_labels = test_generator.classes


# ============================================================
# METRICS
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

recall = recall_score(
    true_labels,
    predictions,
    zero_division=0
)

f1 = f1_score(
    true_labels,
    predictions,
    zero_division=0
)


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 60)
print("TEST RESULTS")
print("=" * 60)

print(f"Test Loss      : {test_loss:.4f}")
print(f"Test Accuracy  : {accuracy * 100:.2f}%")
print(f"Precision      : {precision * 100:.2f}%")
print(f"Recall         : {recall * 100:.2f}%")
print(f"F1-score       : {f1 * 100:.2f}%")


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        true_labels,
        predictions,
        labels=[0, 1],
        target_names=[
            "NORMAL",
            "PNEUMONIA"
        ],
        zero_division=0
    )
)


# ============================================================
# CONFUSION MATRIX
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