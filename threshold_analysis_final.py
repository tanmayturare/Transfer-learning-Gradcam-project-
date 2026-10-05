
import os
import numpy as np
import pandas as pd
import tensorflow as tf

from pathlib import Path
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# =====================================================
# 1. PROJECT PATHS AND SETTINGS
# =====================================================

PROJECT_DIR = Path(r"C:\project\Pneumonia_Detection_Project")

MODEL_PATH = PROJECT_DIR / "models" / "resnet50_final.keras"
DATASET_DIR = PROJECT_DIR / "dataset_final"
RESULTS_DIR = PROJECT_DIR / "results"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

# Class mapping:
# NORMAL = 0
# PNEUMONIA = 1

THRESHOLDS = np.round(np.arange(0.20, 0.801, 0.05), 2)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# =====================================================
# 2. LOAD IMAGE PATHS AND LABELS
# =====================================================

def get_image_paths_and_labels(split_name):
    split_dir = DATASET_DIR / split_name

    image_paths = []
    labels = []

    class_folders = [
        ("NORMAL", 0),
        ("PNEUMONIA", 1)
    ]

    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}

    for class_name, label in class_folders:
        class_dir = split_dir / class_name

        if not class_dir.exists():
            raise FileNotFoundError(
                f"Class folder not found: {class_dir}"
            )

        files = sorted(
            path for path in class_dir.rglob("*")
            if path.is_file()
            and path.suffix.lower() in valid_extensions
        )

        image_paths.extend(files)
        labels.extend([label] * len(files))

    if not image_paths:
        raise ValueError(
            f"No images found in {split_dir}"
        )

    print(f"\n{split_name.upper()} SET")
    print(f"Total images: {len(image_paths)}")
    print(f"NORMAL: {labels.count(0)}")
    print(f"PNEUMONIA: {labels.count(1)}")

    return image_paths, np.array(labels, dtype=np.int32)


# =====================================================
# 3. PREDICT IN BATCHES TO REDUCE MEMORY USAGE
# =====================================================

def predict_probabilities(model, image_paths):
    all_probabilities = []

    for start in range(0, len(image_paths), BATCH_SIZE):
        batch_paths = image_paths[start:start + BATCH_SIZE]
        batch_images = []

        for image_path in batch_paths:
            image = load_img(
                image_path,
                target_size=IMAGE_SIZE,
                color_mode="rgb"
            )

            image_array = img_to_array(image)
            batch_images.append(image_array)

        batch_images = np.array(
            batch_images, dtype=np.float32
        )

        # Use the same ResNet50 preprocessing as training.
        batch_images = preprocess_input(batch_images)

        predictions = model.predict(
            batch_images,
            verbose=0
        )

        probabilities = np.asarray(predictions).reshape(-1)
        all_probabilities.extend(probabilities)

        print(
            f"Processed {min(start + BATCH_SIZE, len(image_paths))}"
            f"/{len(image_paths)} images",
            end="\r"
        )

    print()
    return np.array(all_probabilities, dtype=np.float32)


# =====================================================
# 4. CALCULATE CLASSIFICATION METRICS
# =====================================================

def calculate_metrics(y_true, probabilities, threshold):
    # Probability >= threshold means PNEUMONIA.
    y_pred = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(
        y_true, y_pred, zero_division=0
    )
    pneumonia_recall = recall_score(
        y_true, y_pred, zero_division=0
    )
    normal_recall = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = f1_score(y_true, y_pred, zero_division=0)

    balanced_accuracy = (
        pneumonia_recall + normal_recall
    ) / 2

    return {
        "Threshold": float(threshold),
        "Accuracy": accuracy,
        "Precision": precision,
        "Pneumonia_Recall": pneumonia_recall,
        "Normal_Recall": normal_recall,
        "Balanced_Accuracy": balanced_accuracy,
        "F1_Score": f1,
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp)
    }


# =====================================================
# 5. MAIN PROGRAM
# =====================================================

def main():
    print("Loading saved ResNet50 model...")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    model = load_model(MODEL_PATH)
    print("Model loaded successfully.")

    # -------------------------------------------------
    # Load validation images and predict probabilities
    # -------------------------------------------------

    val_paths, y_val = get_image_paths_and_labels(
        "validation"
    )

    print("\nPredicting validation probabilities...")
    val_probabilities = predict_probabilities(
        model, val_paths
    )

    # -------------------------------------------------
    # Test thresholds using validation data ONLY
    # -------------------------------------------------

    validation_results = []

    for threshold in THRESHOLDS:
        metrics = calculate_metrics(
            y_val, val_probabilities, threshold
        )
        validation_results.append(metrics)

    val_df = pd.DataFrame(validation_results)

    val_csv = RESULTS_DIR / "validation_threshold_results.csv"
    val_df.to_csv(val_csv, index=False)

    print("\n========== VALIDATION THRESHOLD RESULTS ==========")

    display_df = val_df.copy()

    percentage_columns = [
        "Accuracy",
        "Precision",
        "Pneumonia_Recall",
        "Normal_Recall",
        "Balanced_Accuracy",
        "F1_Score"
    ]

    display_df[percentage_columns] = (
        display_df[percentage_columns] * 100
    ).round(2)

    print(display_df.to_string(index=False))

    # Select threshold with highest validation F1 score.
    # In case of a tie, prefer the threshold with higher
    # balanced accuracy.
    best_row = val_df.sort_values(
        by=["F1_Score", "Balanced_Accuracy"],
        ascending=[False, False]
    ).iloc[0]

    best_threshold = float(best_row["Threshold"])

    print("\n========== SELECTED THRESHOLD ==========")
    print(f"Selected using validation set: {best_threshold:.2f}")
    print(f"Validation F1 score: {best_row['F1_Score'] * 100:.2f}%")
    print(
        "Validation balanced accuracy: "
        f"{best_row['Balanced_Accuracy'] * 100:.2f}%"
    )

    # -------------------------------------------------
    # Load test images only after selecting threshold
    # -------------------------------------------------

    test_paths, y_test = get_image_paths_and_labels("test")

    print("\nPredicting test probabilities...")
    test_probabilities = predict_probabilities(
        model, test_paths
    )

    # Evaluate the selected threshold and default 0.50.
    test_thresholds = sorted(set([best_threshold, 0.50]))
    test_results = []

    print("\n========== TEST SET EVALUATION ==========")

    for threshold in test_thresholds:
        metrics = calculate_metrics(
            y_test, test_probabilities, threshold
        )
        test_results.append(metrics)

        y_pred = (
            test_probabilities >= threshold
        ).astype(int)

        print(f"\n--- Threshold: {threshold:.2f} ---")
        print(f"Accuracy: {metrics['Accuracy'] * 100:.2f}%")
        print(f"Precision: {metrics['Precision'] * 100:.2f}%")
        print(
            "PNEUMONIA Recall (sensitivity): "
            f"{metrics['Pneumonia_Recall'] * 100:.2f}%"
        )
        print(
            "NORMAL Recall (specificity): "
            f"{metrics['Normal_Recall'] * 100:.2f}%"
        )
        print(
            f"Balanced Accuracy: "
            f"{metrics['Balanced_Accuracy'] * 100:.2f}%"
        )
        print(f"F1 Score: {metrics['F1_Score'] * 100:.2f}%")

        print("\nConfusion Matrix:")
        print("Rows = actual classes; columns = predicted classes")
        print("                 Predicted NORMAL  Predicted PNEUMONIA")

        tn, fp, fn, tp = (
            metrics["TN"],
            metrics["FP"],
            metrics["FN"],
            metrics["TP"]
        )

        print(f"Actual NORMAL         {tn:5d}              {fp:5d}")
        print(f"Actual PNEUMONIA      {fn:5d}              {tp:5d}")

        print("\nClassification Report:")
        print(
            classification_report(
                y_test,
                y_pred,
                labels=[0, 1],
                target_names=["NORMAL", "PNEUMONIA"],
                zero_division=0,
                digits=4
            )
        )

    # -------------------------------------------------
    # Save test evaluation results
    # -------------------------------------------------

    test_df = pd.DataFrame(test_results)
    test_csv = RESULTS_DIR / "test_threshold_comparison.csv"
    test_df.to_csv(test_csv, index=False)

    # Save chosen threshold and its validation/test metrics.
    chosen_test = test_df[
        test_df["Threshold"] == best_threshold
    ].iloc[0]

    summary = pd.DataFrame([{
        "Model": str(MODEL_PATH.name),
        "Selected_Threshold": best_threshold,
        "Validation_F1": float(best_row["F1_Score"]),
        "Validation_Balanced_Accuracy": float(
            best_row["Balanced_Accuracy"]
        ),
        "Test_Accuracy": float(chosen_test["Accuracy"]),
        "Test_Precision": float(chosen_test["Precision"]),
        "Test_Pneumonia_Recall": float(
            chosen_test["Pneumonia_Recall"]
        ),
        "Test_Normal_Recall": float(
            chosen_test["Normal_Recall"]
        ),
        "Test_F1": float(chosen_test["F1_Score"]),
        "Test_TN": int(chosen_test["TN"]),
        "Test_FP": int(chosen_test["FP"]),
        "Test_FN": int(chosen_test["FN"]),
        "Test_TP": int(chosen_test["TP"])
    }])

    summary_csv = RESULTS_DIR / "final_threshold_summary.csv"
    summary.to_csv(summary_csv, index=False)

    print("\n========== FILES SAVED ==========")
    print(val_csv)
    print(test_csv)
    print(summary_csv)

    print("\nThreshold analysis completed.")


if __name__ == "__main__":
    main()