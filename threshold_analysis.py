import os
import numpy as np
import pandas as pd

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "models/resnet50_finetuned.keras"

TEST_DIR = "dataset/test"

IMAGE_SIZE = (224, 224)

THRESHOLDS = np.arange(
    0.20,
    0.81,
    0.05
)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading model...")

model = load_model(MODEL_PATH)

print("Model loaded successfully!")


# ============================================================
# LOAD TEST DATA
# ============================================================

images = []
true_labels = []
image_names = []


class_names = {
    "NORMAL": 0,
    "PNEUMONIA": 1
}


print("\nLoading test images...")


for class_name, label in class_names.items():

    folder = os.path.join(
        TEST_DIR,
        class_name
    )

    if not os.path.exists(folder):

        print(
            f"WARNING: Folder not found: {folder}"
        )

        continue

    for filename in os.listdir(folder):

        if not filename.lower().endswith(
            (".jpg", ".jpeg", ".png")
        ):
            continue

        image_path = os.path.join(
            folder,
            filename
        )

        try:

            # Load image
            image = load_img(
                image_path,
                target_size=IMAGE_SIZE,
                color_mode="rgb"
            )

            # Convert to numpy
            image_array = img_to_array(
                image
            )

            images.append(
                image_array
            )

            true_labels.append(
                label
            )

            image_names.append(
                filename
            )

        except Exception as e:

            print(
                f"Error loading {image_path}: {e}"
            )


# ============================================================
# CONVERT TO NUMPY
# ============================================================

images = np.array(
    images,
    dtype=np.float32
)

true_labels = np.array(
    true_labels
)


print("\n========================================")
print("TEST DATA INFORMATION")
print("========================================")

print(
    "Total images:",
    len(images)
)

print(
    "NORMAL images:",
    np.sum(true_labels == 0)
)

print(
    "PNEUMONIA images:",
    np.sum(true_labels == 1)
)


# ============================================================
# RESNET50 PREPROCESSING
# ============================================================

print("\nApplying ResNet50 preprocessing...")

processed_images = preprocess_input(
    images
)


# ============================================================
# MODEL PREDICTIONS
# ============================================================

print("\nGenerating predictions...")

predictions = model.predict(
    processed_images,
    batch_size=32,
    verbose=1
)


# Convert predictions to 1D
pneumonia_probabilities = predictions[:, 0]


normal_probabilities = (
    1.0 - pneumonia_probabilities
)


print("\nPredictions generated successfully!")


# ============================================================
# BASIC PROBABILITY ANALYSIS
# ============================================================

print("\n========================================")
print("PROBABILITY ANALYSIS")
print("========================================")


actual_normal = (
    true_labels == 0
)

actual_pneumonia = (
    true_labels == 1
)


print("\nActual NORMAL images:")
print(
    "Average NORMAL probability:",
    f"{normal_probabilities[actual_normal].mean() * 100:.2f}%"
)

print(
    "Average PNEUMONIA probability:",
    f"{pneumonia_probabilities[actual_normal].mean() * 100:.2f}%"
)


print("\nActual PNEUMONIA images:")
print(
    "Average NORMAL probability:",
    f"{normal_probabilities[actual_pneumonia].mean() * 100:.2f}%"
)

print(
    "Average PNEUMONIA probability:",
    f"{pneumonia_probabilities[actual_pneumonia].mean() * 100:.2f}%"
)


# ============================================================
# THRESHOLD ANALYSIS
# ============================================================

results = []


print("\n========================================")
print("THRESHOLD ANALYSIS")
print("========================================")


for threshold in THRESHOLDS:

    # Pneumonia if probability >= threshold
    predicted_labels = (
        pneumonia_probabilities >= threshold
    ).astype(int)


    # Metrics
    accuracy = accuracy_score(
        true_labels,
        predicted_labels
    )

    precision = precision_score(
        true_labels,
        predicted_labels,
        zero_division=0
    )

    recall = recall_score(
        true_labels,
        predicted_labels,
        zero_division=0
    )

    f1 = f1_score(
        true_labels,
        predicted_labels,
        zero_division=0
    )


    # Confusion matrix
    cm = confusion_matrix(
        true_labels,
        predicted_labels
    )

    tn, fp, fn, tp = cm.ravel()


    # NORMAL recall
    normal_recall = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0
    )


    # PNEUMONIA recall
    pneumonia_recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0
    )


    results.append({

        "Threshold": threshold,

        "Accuracy": accuracy,

        "Precision": precision,

        "PNEUMONIA Recall": pneumonia_recall,

        "NORMAL Recall": normal_recall,

        "F1 Score": f1,

        "TN": tn,

        "FP": fp,

        "FN": fn,

        "TP": tp

    })


# ============================================================
# CREATE DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    results
)


# Convert percentages
display_df = results_df.copy()

for column in [
    "Accuracy",
    "Precision",
    "PNEUMONIA Recall",
    "NORMAL Recall",
    "F1 Score"
]:

    display_df[column] = (
        display_df[column] * 100
    ).round(2)


print(
    "\n",
    display_df.to_string(
        index=False
    )
)


# ============================================================
# BEST THRESHOLD BY F1
# ============================================================

best_f1_index = results_df[
    "F1 Score"
].idxmax()


best_f1 = results_df.loc[
    best_f1_index
]


print("\n========================================")
print("BEST THRESHOLD BY F1 SCORE")
print("========================================")

print(
    "Threshold:",
    f"{best_f1['Threshold']:.2f}"
)

print(
    "Accuracy:",
    f"{best_f1['Accuracy'] * 100:.2f}%"
)

print(
    "Precision:",
    f"{best_f1['Precision'] * 100:.2f}%"
)

print(
    "PNEUMONIA Recall:",
    f"{best_f1['PNEUMONIA Recall'] * 100:.2f}%"
)

print(
    "NORMAL Recall:",
    f"{best_f1['NORMAL Recall'] * 100:.2f}%"
)

print(
    "F1 Score:",
    f"{best_f1['F1 Score'] * 100:.2f}%"
)


# ============================================================
# BEST BALANCED THRESHOLD
# ============================================================

results_df["Balanced Recall"] = (
    results_df["NORMAL Recall"]
    + results_df["PNEUMONIA Recall"]
) / 2


best_balanced_index = results_df[
    "Balanced Recall"
].idxmax()


best_balanced = results_df.loc[
    best_balanced_index
]


print("\n========================================")
print("BEST BALANCED RECALL THRESHOLD")
print("========================================")

print(
    "Threshold:",
    f"{best_balanced['Threshold']:.2f}"
)

print(
    "Accuracy:",
    f"{best_balanced['Accuracy'] * 100:.2f}%"
)

print(
    "NORMAL Recall:",
    f"{best_balanced['NORMAL Recall'] * 100:.2f}%"
)

print(
    "PNEUMONIA Recall:",
    f"{best_balanced['PNEUMONIA Recall'] * 100:.2f}%"
)

print(
    "F1 Score:",
    f"{best_balanced['F1 Score'] * 100:.2f}%"
)


# ============================================================
# SAVE RESULTS
# ============================================================

output_file = (
    "results/threshold_analysis.csv"
)

os.makedirs(
    "results",
    exist_ok=True
)

display_df.to_csv(
    output_file,
    index=False
)


print(
    f"\nResults saved to: {output_file}"
)


# ============================================================
# DETAILED RESULT AT 0.50
# ============================================================

threshold_050 = results_df[
    np.isclose(
        results_df["Threshold"],
        0.50
    )
].iloc[0]


print("\n========================================")
print("CURRENT THRESHOLD = 0.50")
print("========================================")

print(
    "Accuracy:",
    f"{threshold_050['Accuracy'] * 100:.2f}%"
)

print(
    "NORMAL Recall:",
    f"{threshold_050['NORMAL Recall'] * 100:.2f}%"
)

print(
    "PNEUMONIA Recall:",
    f"{threshold_050['PNEUMONIA Recall'] * 100:.2f}%"
)

print(
    "F1 Score:",
    f"{threshold_050['F1 Score'] * 100:.2f}%"
)

print(
    "\nConfusion Matrix:"
)

print(
    f"""
                Predicted
              NORMAL  PNEUMONIA

Actual NORMAL    {int(threshold_050['TN']):3d}       {int(threshold_050['FP']):3d}

Actual PNEUMONIA {int(threshold_050['FN']):3d}       {int(threshold_050['TP']):3d}
"""
)


print("\nAnalysis complete!")