import os
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D



TRAIN_DIR = "dataset/train"
VALIDATION_DIR = "dataset/validation"
TEST_DIR = "dataset/test"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

MODEL_PATH = "models/resnet50_pneumonia.keras"




train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=10,
    zoom_range=0.1,
    horizontal_flip=True
)

validation_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0
)

test_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0
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




base_model.trainable = False




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
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)




model.summary()




EPOCHS = 5

history = model.fit(
    train_data,
    validation_data=validation_data,
    epochs=EPOCHS
)




os.makedirs(
    "models",
    exist_ok=True
)

model.save(
    MODEL_PATH
)

print("\nResNet50 model saved successfully.")
print("Saved at:", MODEL_PATH)




os.makedirs(
    "results",
    exist_ok=True
)

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
    "ResNet50 Transfer Learning Accuracy"
)

plt.legend()

plt.savefig(
    "results/resnet50_accuracy.png",
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
    "ResNet50 Transfer Learning Loss"
)

plt.legend()

plt.savefig(
    "results/resnet50_loss.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()




print("\n")
print("=" * 50)
print("RESNET50 TRAINING COMPLETED")
print("=" * 50)

print("Model:", MODEL_PATH)
print("Accuracy graph: results/resnet50_accuracy.png")
print("Loss graph: results/resnet50_loss.png")