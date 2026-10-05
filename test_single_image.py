import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input


MODEL_PATH = "models/resnet50_finetuned.keras"
IMAGE_PATH = "dataset/test/NORMAL/NORMAL2-IM-0369-0001.jpeg"

IMAGE_SIZE = (224, 224)


# Load model
model = load_model(MODEL_PATH)

# Load image
image = Image.open(IMAGE_PATH)

print("Original image mode:", image.mode)
print("Original image size:", image.size)

# Convert to RGB
image = image.convert("RGB")

# Resize
image = image.resize(IMAGE_SIZE)

# Convert to numpy
image_array = np.array(image).astype(np.float32)

print("Before preprocessing:")
print("Min:", image_array.min())
print("Max:", image_array.max())
print("Shape:", image_array.shape)

# Add batch dimension
image_array = np.expand_dims(image_array, axis=0)

# ResNet50 preprocessing
processed_image = preprocess_input(image_array)

print("\nAfter preprocessing:")
print("Min:", processed_image.min())
print("Max:", processed_image.max())

# Prediction
prediction = model.predict(
    processed_image,
    verbose=0
)

pneumonia_probability = float(prediction[0][0])
normal_probability = 1 - pneumonia_probability

print("\n==============================")
print("RESULT")
print("==============================")

print(
    f"Normal Probability: "
    f"{normal_probability * 100:.2f}%"
)

print(
    f"Pneumonia Probability: "
    f"{pneumonia_probability * 100:.2f}%"
)

if pneumonia_probability >= 0.5:

    print("Prediction: PNEUMONIA")

else:

    print("Prediction: NORMAL")

print("==============================")