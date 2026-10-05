import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input




MODEL_PATH = "models/resnet50_finetuned.keras"

IMAGE_PATH = "sample_xray.jpg"

IMAGE_SIZE = (224, 224)

RESULTS_DIR = "results"

LAST_CONV_LAYER = "conv5_block3_out"




os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)




if not os.path.exists(MODEL_PATH):

    raise FileNotFoundError(
        f"\nModel not found:\n{MODEL_PATH}\n"
        "Make sure the model exists in the models folder."
    )


print("\n" + "=" * 60)
print("LOADING RESNET50 MODEL")
print("=" * 60)

model = load_model(MODEL_PATH)

print("\nFine-tuned ResNet50 loaded successfully.")




print("\nLooking for Grad-CAM layer...")

try:

    last_conv_layer = model.get_layer(
        LAST_CONV_LAYER
    )

except ValueError:

    print("\nAvailable convolutional layers:")

    for layer in model.layers:

        if isinstance(
            layer,
            tf.keras.layers.Conv2D
        ):

            print(layer.name)

    raise ValueError(
        f"\nLayer '{LAST_CONV_LAYER}' was not found."
    )


print(
    f"Using Grad-CAM layer: "
    f"{LAST_CONV_LAYER}"
)




grad_model = tf.keras.Model(
    inputs=model.inputs,
    outputs=[
        last_conv_layer.output,
        model.output
    ]
)




if not os.path.exists(IMAGE_PATH):

    raise FileNotFoundError(
        f"\nImage not found:\n{IMAGE_PATH}\n\n"
        "Place your sample X-ray image in the "
        "project folder."
    )


print("\n" + "=" * 60)
print("LOADING X-RAY")
print("=" * 60)

original_image = Image.open(
    IMAGE_PATH
).convert("RGB")

print(
    f"\nOriginal image size: "
    f"{original_image.size}"
)




resized_image = original_image.resize(
    IMAGE_SIZE
)




image_array = np.array(
    resized_image,
    dtype=np.float32
)



image_array = preprocess_input(
    image_array
)

input_tensor = np.expand_dims(
    image_array,
    axis=0
)




print("\nCalculating Grad-CAM...")

with tf.GradientTape() as tape:

    conv_outputs, predictions = grad_model(
        input_tensor,
        training=False
    )

    
    pneumonia_probability = predictions[:, 0]

    
    normal_probability = (
        1.0 - pneumonia_probability
    )

    
    predicted_class_index = tf.cast(
        pneumonia_probability[0] >= 0.65,
        tf.int32
    )

    
    class_score = tf.where(
        predicted_class_index == 1,
        pneumonia_probability[0],
        normal_probability[0]
    )




gradients = tape.gradient(
    class_score,
    conv_outputs
)


if gradients is None:

    raise ValueError(
        "\nGradients are unavailable.\n"
        "Check that the selected convolutional "
        "layer is connected to the model output."
    )




pooled_gradients = tf.reduce_mean(
    gradients,
    axis=(0, 1, 2)
)




feature_maps = conv_outputs[0]




heatmap = tf.reduce_sum(
    feature_maps * pooled_gradients,
    axis=-1
)




heatmap = tf.maximum(
    heatmap,
    0
)


# Normalize heatmap

max_value = tf.reduce_max(
    heatmap
)

heatmap = tf.where(
    max_value > 0,
    heatmap / (max_value + 1e-8),
    tf.zeros_like(heatmap)
)


heatmap = heatmap.numpy()




pneumonia_probability = float(
    predictions.numpy()[0][0]
)

normal_probability = (
    1.0 - pneumonia_probability
)


if pneumonia_probability >= 0.65:

    predicted_class = "PNEUMONIA"

    confidence = (
        pneumonia_probability * 100
    )

else:

    predicted_class = "NORMAL"

    confidence = (
        normal_probability * 100
    )




print("\n" + "=" * 60)
print("PREDICTION")
print("=" * 60)

print(
    f"\nPredicted Class      : "
    f"{predicted_class}"
)

print(
    f"NORMAL Probability   : "
    f"{normal_probability * 100:.2f}%"
)

print(
    f"PNEUMONIA Probability: "
    f"{pneumonia_probability * 100:.2f}%"
)

print(
    f"Confidence           : "
    f"{confidence:.2f}%"
)




heatmap_image = Image.fromarray(
    np.uint8(255 * heatmap)
).resize(
    IMAGE_SIZE
)


heatmap_array = (
    np.array(
        heatmap_image,
        dtype=np.float32
    )
    / 255.0
)




colored_heatmap = plt.get_cmap(
    "jet"
)(
    heatmap_array
)[..., :3]




display_image = (
    np.array(
        resized_image,
        dtype=np.float32
    )
    / 255.0
)




overlay = (
    0.6 * display_image
    +
    0.4 * colored_heatmap
)


overlay = np.clip(
    overlay,
    0,
    1
)




heatmap_path = os.path.join(
    RESULTS_DIR,
    "resnet50_gradcam_heatmap.png"
)


plt.figure(
    figsize=(6, 6)
)

plt.imshow(
    heatmap_array,
    cmap="jet"
)

plt.title(
    f"Grad-CAM Heatmap - {predicted_class}"
)

plt.axis("off")

plt.colorbar(
    fraction=0.046,
    pad=0.04
)

plt.tight_layout()

plt.savefig(
    heatmap_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()




overlay_path = os.path.join(
    RESULTS_DIR,
    "resnet50_gradcam_overlay.png"
)


plt.figure(
    figsize=(6, 6)
)

plt.imshow(
    overlay
)

plt.title(
    f"{predicted_class} - "
    f"{confidence:.2f}% Confidence"
)

plt.axis("off")

plt.tight_layout()

plt.savefig(
    overlay_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()




final_path = os.path.join(
    RESULTS_DIR,
    "resnet50_gradcam.png"
)


plt.figure(
    figsize=(15, 5)
)




plt.subplot(
    1,
    3,
    1
)

plt.imshow(
    display_image
)

plt.title(
    "Original X-ray"
)

plt.axis("off")




plt.subplot(
    1,
    3,
    2
)

plt.imshow(
    heatmap_array,
    cmap="jet"
)

plt.title(
    "Grad-CAM Heatmap"
)

plt.axis("off")




plt.subplot(
    1,
    3,
    3
)

plt.imshow(
    overlay
)

plt.title(
    f"{predicted_class}\n"
    f"Confidence: {confidence:.2f}%"
)

plt.axis("off")


plt.tight_layout()


plt.savefig(
    final_path,
    dpi=300,
    bbox_inches="tight"
)


plt.show()




print("\n" + "=" * 60)
print("GRAD-CAM COMPLETED")
print("=" * 60)

print("\nPrediction:")
print(
    f"Class      : {predicted_class}"
)

print(
    f"Confidence : {confidence:.2f}%"
)

print("\nSaved files:")

print(
    f"\n1. {heatmap_path}"
)

print(
    f"2. {overlay_path}"
)

print(
    f"3. {final_path}"
)

print("\n" + "=" * 60)