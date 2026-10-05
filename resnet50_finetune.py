import os
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
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




TRAIN_DIR = "dataset/train"
VALIDATION_DIR = "dataset/validation"
TEST_DIR = "dataset/test"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

MODEL_PATH = "models/resnet50_finetuned.keras"




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




print("\nClass mapping:")
print(train_data.class_indices)




base_model = ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)




base_model.trainable = True

# Freeze the early layers
for layer in base_model.layers[:140]:
    layer.trainable = False




print("\nResNet50 layer status:")
print("======================")

for layer in base_model.layers:
    if layer.trainable:
        print(layer.name, "-> Trainable")




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




early_stop = EarlyStopping(
    monitor="val_loss",
    patience=2,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    MODEL_PATH,
    monitor="val_loss",
    save_best_only=True
)




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




model = load_model(
    MODEL_PATH
)




test_loss, test_accuracy = model.evaluate(
    test_data,
    verbose=1
)




test_data.reset()

predictions = model.predict(
    test_data,
    verbose=1
)

y_pred = (
    predictions > 0.5
).astype(int).flatten()

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
print("=" * 55)
print("RESNET50 FINE-TUNING TEST RESULTS")
print("=" * 55)

print(f"Test Loss      : {test_loss:.4f}")
print(f"Test Accuracy  : {accuracy * 100:.2f}%")
print(f"Precision      : {precision * 100:.2f}%")
print(f"Recall         : {recall * 100:.2f}%")
print(f"F1-score       : {f1 * 100:.2f}%")




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




cm = confusion_matrix(
    y_true,
    y_pred
)

print("\n")
print("=" * 55)
print("CONFUSION MATRIX")
print("=" * 55)

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

plt.title(
    "ResNet50 Fine-Tuned Confusion Matrix"
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
    "results/resnet50_finetuned_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()




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
    "ResNet50 Fine-Tuning Accuracy"
)

plt.legend()

plt.savefig(
    "results/resnet50_finetuned_accuracy.png",
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
    "ResNet50 Fine-Tuning Loss"
)

plt.legend()

plt.savefig(
    "results/resnet50_finetuned_loss.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()




print("\n")
print("=" * 55)
print("RESNET50 FINE-TUNING COMPLETED")
print("=" * 55)

print(
    "Best model saved at:"
)

print(
    MODEL_PATH
)

print(
    "\nResults saved in the results folder."
)