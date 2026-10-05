import os
import numpy as np
import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt

from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "models/resnet50_finetuned.keras"
IMAGE_SIZE = (224, 224)
LAST_CONV_LAYER = "conv5_block3_out"


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Pneumonia Detection System",
    page_icon="🫁",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        text-align: center;
        font-size: 40px;
        font-weight: bold;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #666666;
        margin-bottom: 30px;
    }

    .result-normal {
        padding: 20px;
        border-radius: 12px;
        background-color: #dff5e1;
        border: 2px solid #4caf50;
        text-align: center;
        font-size: 28px;
        font-weight: bold;
    }

    .result-pneumonia {
        padding: 20px;
        border-radius: 12px;
        background-color: #ffe1e1;
        border: 2px solid #e53935;
        text-align: center;
        font-size: 28px;
        font-weight: bold;
    }

    .info-box {
        padding: 15px;
        border-radius: 10px;
        background-color: #f5f5f5;
        margin-top: 15px;
        margin-bottom: 15px;
    }

    .disclaimer {
        padding: 15px;
        border-radius: 10px;
        background-color: #fff4d6;
        border: 1px solid #e0b84c;
        margin-top: 25px;
        font-size: 14px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="main-title">🫁 Pneumonia Detection System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Chest X-ray Classification using Fine-Tuned ResNet50'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_pneumonia_model():

    if not os.path.exists(MODEL_PATH):
        return None

    model = load_model(MODEL_PATH)

    return model


model = load_pneumonia_model()


# ============================================================
# MODEL ERROR CHECK
# ============================================================

if model is None:

    st.error(
        f"Model not found at:\n\n`{MODEL_PATH}`\n\n"
        "Make sure the model file exists in the `models` folder."
    )

    st.stop()


# ============================================================
# GRAD-CAM FUNCTION
# ============================================================

def make_gradcam(model, image_array, predicted_class):
    """
    Generate Grad-CAM heatmap for the predicted class.

    predicted_class:
        0 = NORMAL
        1 = PNEUMONIA
    """

    try:

        # Get the last convolutional layer
        last_conv_layer = model.get_layer(LAST_CONV_LAYER)

    except ValueError:

        st.error(
            f"Could not find Grad-CAM layer: "
            f"{LAST_CONV_LAYER}"
        )

        return None

    # Create a model that outputs:
    # 1. feature maps from last convolutional layer
    # 2. final prediction
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            last_conv_layer.output,
            model.output
        ]
    )

    # Convert image to TensorFlow tensor
    image_tensor = tf.cast(image_array, tf.float32)

    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(image_tensor)

        # Model output is pneumonia probability
        pneumonia_probability = predictions[:, 0]

        # Normal probability
        normal_probability = 1.0 - pneumonia_probability

        if predicted_class == 1:

            class_score = pneumonia_probability

        else:

            class_score = normal_probability

    # Calculate gradients
    gradients = tape.gradient(
        class_score,
        conv_outputs
    )

    # Global average pooling of gradients
    pooled_gradients = tf.reduce_mean(
        gradients,
        axis=(1, 2)
    )

    # Remove batch dimension
    conv_outputs = conv_outputs[0]

    pooled_gradients = pooled_gradients[0]

    # Weight feature maps using gradients
    heatmap = tf.reduce_sum(
        conv_outputs * pooled_gradients,
        axis=-1
    )

    # Keep positive influence
    heatmap = tf.maximum(heatmap, 0)

    # Normalize between 0 and 1
    max_value = tf.reduce_max(heatmap)

    if max_value > 0:
        heatmap /= max_value

    return heatmap.numpy()


# ============================================================
# PREPARE IMAGE
# ============================================================

def prepare_image(image):

    # Convert to RGB
    image = image.convert("RGB")

    # Resize
    resized_image = image.resize(IMAGE_SIZE)

    # Convert to numpy array
    image_array = np.array(resized_image).astype(np.float32)

    # Add batch dimension
    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # ResNet50 preprocessing
    processed_image = preprocess_input(
        image_array
    )

    return resized_image, processed_image


# ============================================================
# CREATE GRAD-CAM OVERLAY
# ============================================================

def create_gradcam_overlay(original_image, heatmap):

    # Convert original image to numpy
    original_array = np.array(
        original_image
    )

    # Create figure
    fig, ax = plt.subplots(
        figsize=(7, 7)
    )

    # Display original X-ray
    ax.imshow(original_array)

    # Display heatmap
    ax.imshow(
        heatmap,
        cmap="jet",
        alpha=0.45,
        extent=(
            0,
            original_array.shape[1],
            original_array.shape[0],
            0
        )
    )

    ax.axis("off")

    fig.tight_layout(
        pad=0
    )

    return fig


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("About the Model")

    st.write(
        """
        This application uses a **fine-tuned ResNet50**
        convolutional neural network for classifying
        chest X-ray images.
        """
    )

    st.write("**Input Size:** 224 × 224 pixels")

    st.write("**Classes:**")

    st.write("- NORMAL")
    st.write("- PNEUMONIA")

    st.write("**Model:** Fine-Tuned ResNet50")

    st.write("**Grad-CAM Layer:**")
    st.code(LAST_CONV_LAYER)

    st.divider()

    st.subheader("Model Performance")

    st.metric(
        "Test Accuracy",
        "91.19%"
    )

    st.metric(
        "F1 Score",
        "93.18%"
    )

    st.divider()

    st.write(
        "Upload a chest X-ray image to obtain "
        "a model prediction and Grad-CAM visualization."
    )


# ============================================================
# IMAGE UPLOAD
# ============================================================

st.subheader("Upload Chest X-ray")

uploaded_file = st.file_uploader(
    "Choose a chest X-ray image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


# ============================================================
# PROCESS IMAGE
# ============================================================

if uploaded_file is not None:

    # Open uploaded image
    image = Image.open(uploaded_file)

    # Prepare image
    resized_image, processed_image = prepare_image(
        image
    )

    # --------------------------------------------------------
    # DISPLAY ORIGINAL IMAGE
    # --------------------------------------------------------

    st.subheader("Uploaded X-ray")

    col1, col2 = st.columns(2)

    with col1:

        st.image(
            image,
            caption="Original X-ray",
            use_container_width=True
        )

    with col2:

        st.image(
            resized_image,
            caption="Image used for prediction (224 × 224)",
            use_container_width=True
        )


    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    with st.spinner(
        "Analyzing X-ray..."
    ):

        prediction = model.predict(
            processed_image,
            verbose=0
        )

    # Model output = pneumonia probability
    pneumonia_probability = float(
        prediction[0][0]
    )

    # Normal probability
    normal_probability = (
        1.0 - pneumonia_probability
    )


    # --------------------------------------------------------
    # CLASSIFICATION
    # --------------------------------------------------------

    if pneumonia_probability >= 0.65:

        predicted_class = 1
        predicted_label = "PNEUMONIA"
        confidence = pneumonia_probability

    else:

        predicted_class = 0
        predicted_label = "NORMAL"
        confidence = normal_probability


    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    st.subheader("Prediction Result")

    if predicted_class == 1:

        st.markdown(
            f"""
            <div class="result-pneumonia">
                ⚠️ PNEUMONIA DETECTED
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            f"""
            <div class="result-normal">
                ✅ NORMAL
            </div>
            """,
            unsafe_allow_html=True
        )


    # --------------------------------------------------------
    # CONFIDENCE
    # --------------------------------------------------------

    st.write("")

    result_col1, result_col2 = st.columns(2)

    with result_col1:

        st.metric(
            "Predicted Class",
            predicted_label
        )

    with result_col2:

        st.metric(
            "Confidence",
            f"{confidence * 100:.2f}%"
        )


    # --------------------------------------------------------
    # PROBABILITY DETAILS
    # --------------------------------------------------------

    st.subheader("Prediction Probabilities")

    probability_col1, probability_col2 = st.columns(2)

    with probability_col1:

        st.metric(
            "NORMAL Probability",
            f"{normal_probability * 100:.2f}%"
        )

    with probability_col2:

        st.metric(
            "PNEUMONIA Probability",
            f"{pneumonia_probability * 100:.2f}%"
        )


    # --------------------------------------------------------
    # PROBABILITY BAR
    # --------------------------------------------------------

    st.write("### Probability Distribution")

    probability_data = np.array(
        [
            normal_probability,
            pneumonia_probability
        ]
    )

    st.bar_chart(
        {
            "NORMAL": [normal_probability],
            "PNEUMONIA": [pneumonia_probability]
        }
    )


    # --------------------------------------------------------
    # GRAD-CAM
    # --------------------------------------------------------

    st.subheader("Grad-CAM Visualization")

    st.info(
        "Grad-CAM highlights image regions that contributed "
        "to the model's prediction. It should be interpreted "
        "as a model-explanation visualization, not as proof "
        "of disease."
    )

    with st.spinner(
        "Generating Grad-CAM..."
    ):

        heatmap = make_gradcam(
            model,
            processed_image,
            predicted_class
        )


    if heatmap is not None:

        # Display heatmap
        gradcam_col1, gradcam_col2 = st.columns(2)

        with gradcam_col1:

            st.write("### Grad-CAM Heatmap")

            st.image(
                heatmap,
                caption="Grad-CAM Heatmap",
                use_container_width=True
            )

        with gradcam_col2:

            st.write("### Grad-CAM Overlay")

            fig = create_gradcam_overlay(
                resized_image,
                heatmap
            )

            st.pyplot(
                fig,
                clear_figure=True
            )


    # --------------------------------------------------------
    # TECHNICAL INFORMATION
    # --------------------------------------------------------

    with st.expander(
        "View Technical Details"
    ):

        st.write(
            "**Model:** ResNet50"
        )

        st.write(
            "**Transfer Learning:** ImageNet"
        )

        st.write(
            "**Fine-Tuning:** Yes"
        )

        st.write(
            "**Input Size:** 224 × 224"
        )

        st.write(
            "**Preprocessing:** "
            "ResNet50 preprocess_input"
        )

        st.write(
            "**Output Activation:** Sigmoid"
        )

        st.write(
            "**Classification Threshold:** 0.65"
        )

        st.write(
            "**Grad-CAM Layer:** "
            f"{LAST_CONV_LAYER}"
        )


# ============================================================
# DISCLAIMER
# ============================================================

st.markdown(
    """
    <div class="disclaimer">
    <b>⚠️ Academic Project Disclaimer:</b><br><br>

    This system is developed as an academic demonstration
    of deep-learning-based chest X-ray image classification.
    It is not a clinically validated medical diagnostic system
    and should not be used as a substitute for professional
    medical examination or diagnosis.
    </div>
    """,
    unsafe_allow_html=True
)