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
    "resnet50_final.keras"
)

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)




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
    shuffle=True
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




base_model = ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)


# Enable fine-tuning
base_model.trainable = True


# Freeze first 140 layers
for layer in base_model.layers[:140]:
    layer.trainable = False




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




model.compile(
    optimizer=Adam(
        learning_rate=0.00001
    ),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)




print("\nModel Summary:")
model.summary()




early_stop = EarlyStopping(
    monitor="val_loss",
    patience=2,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    MODEL_PATH,
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)




print("\n")
print("=" * 60)
print("STARTING FINAL RESNET50 FINE-TUNING")
print("=" * 60)

history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=EPOCHS,
    callbacks=[
        early_stop,
        checkpoint
    ]
)




print("\nLoading best model...")

model = load_model(MODEL_PATH)




print("\n")
print("=" * 60)
print("FINAL TEST EVALUATION")
print("=" * 60)

test_loss, test_accuracy = model.evaluate(
    test_generator,
    verbose=1
)




test_generator.reset()

probabilities = model.predict(
    test_generator,
    verbose=1
)

predictions = (
    probabilities > 0.5
).astype(int).flatten()

true_labels = test_generator.classes




accuracy = accuracy_score(
    true_labels,
    predictions
)

precision = precision_score(
    true_labels,
    predictions
)

recall = recall_score(
    true_labels,
    predictions
)

f1 = f1_score(
    true_labels,
    predictions
)




print("\n")
print("=" * 60)
print("FINAL RESNET50 TEST RESULTS")
print("=" * 60)

print(f"Test Loss      : {test_loss:.4f}")
print(f"Test Accuracy  : {accuracy * 100:.2f}%")
print(f"Precision      : {precision * 100:.2f}%")
print(f"Recall         : {recall * 100:.2f}%")
print(f"F1-score       : {f1 * 100:.2f}%")




print("\n")
print("=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        true_labels,
        predictions,
        target_names=[
            "NORMAL",
            "PNEUMONIA"
        ]
    )
)




cm = confusion_matrix(
    true_labels,
    predictions
)

print("\n")
print("=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print(cm)




plt.figure(figsize=(6, 5))

plt.imshow(cm)

plt.title(
    "ResNet50 Final - Confusion Matrix"
)

plt.colorbar()

plt.xticks(
    [0, 1],
    ["NORMAL", "PNEUMONIA"]
)

plt.yticks(
    [0, 1],
    ["NORMAL", "PNEUMONIA"]
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")

for i in range(2):
    for j in range(2):
        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "resnet50_final_confusion_matrix.png"
    ),
    dpi=300
)

plt.close()




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
    "ResNet50 Final - Accuracy"
)

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "resnet50_final_accuracy.png"
    ),
    dpi=300
)

plt.close()




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
    "ResNet50 Final - Loss"
)

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "resnet50_final_loss.png"
    ),
    dpi=300
)

plt.close()




print("\n")
print("=" * 60)
print("FINAL MODEL SAVED")
print("=" * 60)

print(
    f"Model: {MODEL_PATH}"
)

print(
    "Confusion Matrix saved."
)

print(
    "Accuracy graph saved."
)

print(
    "Loss graph saved."
)

print("\nTraining and evaluation completed.")