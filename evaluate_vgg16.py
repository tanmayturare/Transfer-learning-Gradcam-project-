import os
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)




TEST_DIR = "dataset/test"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

MODEL_PATH = "models/vgg16_pneumonia.keras"




test_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0
)

test_data = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)




print("\nClass mapping:")
print(test_data.class_indices)




print("\nLoading VGG16 model...")

model = load_model(MODEL_PATH)

print("VGG16 model loaded successfully.")




test_loss, test_accuracy = model.evaluate(
    test_data,
    verbose=1
)




test_data.reset()

predictions = model.predict(
    test_data,
    verbose=1
)

# Convert probabilities into class predictions
y_pred = (predictions > 0.5).astype(int).flatten()

# Actual class labels
y_true = test_data.classes




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




print("\n")
print("=" * 50)
print("VGG16 TEST RESULTS")
print("=" * 50)

print(f"Test Loss      : {test_loss:.4f}")
print(f"Test Accuracy  : {accuracy * 100:.2f}%")
print(f"Precision      : {precision * 100:.2f}%")
print(f"Recall         : {recall * 100:.2f}%")
print(f"F1-score       : {f1 * 100:.2f}%")




class_names = list(test_data.class_indices.keys())

print("\n")
print("=" * 50)
print("CLASSIFICATION REPORT")
print("=" * 50)

print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0
    )
)




cm = confusion_matrix(
    y_true,
    y_pred
)

print("\n")
print("=" * 50)
print("CONFUSION MATRIX")
print("=" * 50)

print(cm)




os.makedirs(
    "results",
    exist_ok=True
)

plt.figure(figsize=(7, 6))

plt.imshow(
    cm,
    interpolation="nearest"
)

plt.title("VGG16 Confusion Matrix")

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

# Write numbers inside confusion matrix
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
    "results/vgg16_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()




print("\n")
print("Evaluation completed successfully.")

print(
    "Confusion matrix saved to: "
    "results/vgg16_confusion_matrix.png"
)