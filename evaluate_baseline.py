
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)



MODEL_PATH = "models/baseline_cnn.keras"
TEST_DIR = "dataset/test"
RESULTS_DIR = "results"

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
THRESHOLD = 0.5

os.makedirs(RESULTS_DIR, exist_ok=True)



if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}. "
        "Train the baseline CNN first."
    )

if not os.path.isdir(TEST_DIR):
    raise FileNotFoundError(
        f"Test folder not found: {TEST_DIR}. "
        "Check your dataset folder structure."
    )



print("\nLoading trained baseline CNN...")

model = load_model(MODEL_PATH)

print("Model loaded successfully.")



test_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0
)

test_data = test_datagen.flow_from_directory(
    directory=TEST_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

print("\nClass mapping:", test_data.class_indices)
print("Number of test images:", test_data.samples)


expected_classes = {"NORMAL", "PNEUMONIA"}

if set(test_data.class_indices.keys()) != expected_classes:
    raise ValueError(
        "Expected NORMAL and PNEUMONIA folders. "
        f"Found: {test_data.class_indices}"
    )

if test_data.samples == 0:
    raise ValueError("No test images were found.")



print("\nEvaluating model on test images...")

test_loss, test_accuracy = model.evaluate(
    test_data,
    verbose=1
)

print(f"\nTest Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")



test_data.reset()

probabilities = model.predict(
    test_data,
    verbose=1
).ravel()

predicted_labels = (
    probabilities >= THRESHOLD
).astype(int)

true_labels = test_data.classes


pneumonia_index = test_data.class_indices["PNEUMONIA"]

if pneumonia_index != 1:
    raise ValueError(
        "This script expects PNEUMONIA to be class 1. "
        f"Actual mapping: {test_data.class_indices}"
    )



accuracy = accuracy_score(
    true_labels, predicted_labels
)

precision = precision_score(
    true_labels, predicted_labels,
    zero_division=0
)

recall = recall_score(
    true_labels, predicted_labels,
    zero_division=0
)

f1 = f1_score(
    true_labels, predicted_labels,
    zero_division=0
)

print("\n========== BASELINE CNN RESULTS ==========")
print(f"Accuracy:  {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall:    {recall * 100:.2f}%")
print(f"F1-score:  {f1 * 100:.2f}%")



print("\n========== CLASSIFICATION REPORT ==========")

report = classification_report(
    true_labels,
    predicted_labels,
    labels=[0, 1],
    target_names=["NORMAL", "PNEUMONIA"],
    zero_division=0
)

print(report)



cm = confusion_matrix(
    true_labels,
    predicted_labels,
    labels=[0, 1]
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["NORMAL", "PNEUMONIA"]
)

display.plot(
    cmap="Blues",
    values_format="d"
)

plt.title("Baseline CNN - Confusion Matrix")
plt.tight_layout()

confusion_matrix_path = os.path.join(
    RESULTS_DIR,
    "baseline_cnn_confusion_matrix.png"
)

plt.savefig(
    confusion_matrix_path,
    dpi=300,
    bbox_inches="tight"
)

print(
    "\nConfusion matrix saved to:",
    confusion_matrix_path
)

plt.show()



results_path = os.path.join(
    RESULTS_DIR,
    "baseline_cnn_evaluation.txt"
)

with open(results_path, "w", encoding="utf-8") as file:
    file.write("BASELINE CNN EVALUATION RESULTS\n")
    file.write("================================\n")
    file.write(f"Test images: {test_data.samples}\n")
    file.write(f"Test loss: {test_loss:.4f}\n")
    file.write(f"Test accuracy: {accuracy * 100:.2f}%\n")
    file.write(f"Precision: {precision * 100:.2f}%\n")
    file.write(f"Recall: {recall * 100:.2f}%\n")
    file.write(f"F1-score: {f1 * 100:.2f}%\n\n")
    file.write("CLASSIFICATION REPORT\n")
    file.write(report)
    file.write("\nCONFUSION MATRIX\n")
    file.write(str(cm))
    file.write("\n")

print("Evaluation report saved to:", results_path)
print("\nBaseline CNN evaluation completed successfully.")