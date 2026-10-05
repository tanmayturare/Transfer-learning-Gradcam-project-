import tensorflow as tf
import os
import matplotlib.pyplot as plt


from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout


TRAIN_DIR = "dataset/train"
VALIDATION_DIR = "dataset/validation"
TEST_DIR = "dataset/test"


IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32


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


model = Sequential()

model.add(
    Conv2D(
        32,
        (3, 3),
        activation="relu",
        input_shape=(128, 128, 3)
    )
)

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(
    Conv2D(
        64,
        (3, 3),
        activation="relu"
    )
)

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(
    Conv2D(
        128,
        (3, 3),
        activation="relu"
    )
)

model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Flatten())


model.add(
    Dense(
        128,
        activation="relu"
    )
)


model.add(
    Dropout(0.5)
)



model.add(
    Dense(
        1,
        activation="sigmoid"
    )
)



model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


model.summary()


EPOCHS = 10

history = model.fit(
    train_data,
    validation_data=validation_data,
    epochs=EPOCHS
)



os.makedirs("models", exist_ok=True)

model.save(
    "models/baseline_cnn.keras"
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
plt.title("Baseline CNN Accuracy")

plt.legend()

os.makedirs("results", exist_ok=True)

plt.savefig(
    "results/baseline_cnn_accuracy.png",
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
plt.title("Baseline CNN Loss")

plt.legend()


plt.legend()

plt.savefig(
    "results/baseline_cnn_loss.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

plt.show()



