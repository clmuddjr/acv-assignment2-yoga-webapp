import json
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf


# ---------------------------------------------------------
# STREAMLIT PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="Yoga Pose Classifier",
    page_icon="🧘",
    layout="centered"
)


# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------
APP_ROOT = Path(__file__).resolve().parent
MODELS_DIR = APP_ROOT / "models"
ASSETS_DIR = APP_ROOT / "assets"
CLASS_NAMES_PATH = ASSETS_DIR / "class_names.json"

IMAGE_SIZE = (128, 128)


# ---------------------------------------------------------
# LOAD CLASS NAMES
# ---------------------------------------------------------
with open(CLASS_NAMES_PATH, "r") as f:
    CLASS_NAMES = json.load(f)


# ---------------------------------------------------------
# AVAILABLE MODELS
# ---------------------------------------------------------
MODEL_OPTIONS = {
    "Assignment 1 Baseline CNN": MODELS_DIR / "Assignment1_Baseline_CNN.tflite",
    "EfficientNetB0 Frozen Only": MODELS_DIR / "EfficientNetB0_FrozenOnly.tflite",
    "ResNet50 Frozen Only": MODELS_DIR / "ResNet50_FrozenOnly.tflite",
    "MobileNetV2 Frozen Only": MODELS_DIR / "MobileNetV2_FrozenOnly.tflite",
}


# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------
@st.cache_resource
def load_tflite_interpreter(model_path: str):
    """
    Load and allocate a TFLite interpreter.
    """
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter


def preprocess_image(uploaded_image: Image.Image) -> np.ndarray:
    """
    Prepare an uploaded image for TFLite inference.

    The models were trained on 128x128 RGB images. Because the transfer-learning
    models include preprocessing inside the saved model graph, the app only needs
    to resize the image and convert it to float32.
    """
    image = uploaded_image.convert("RGB")
    image = image.resize(IMAGE_SIZE)
    image_array = np.array(image, dtype=np.float32)

    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)
    return image_array


def run_tflite_inference(interpreter, input_data: np.ndarray) -> np.ndarray:
    """
    Run inference on a TFLite model and return the predicted probabilities.
    """
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]["index"])

    return output_data[0]


def get_top_k_predictions(probabilities: np.ndarray, class_names: list, k: int = 3):
    """
    Return the top-k predictions as (class_name, probability) pairs.
    """
    top_indices = np.argsort(probabilities)[::-1][:k]
    return [(class_names[i], float(probabilities[i])) for i in top_indices]


# ---------------------------------------------------------
# STREAMLIT UI
# ---------------------------------------------------------
st.title("🧘 Yoga Pose Classification Web Application")
st.write(
    "Upload a yoga image, select one of the quantized model versions, "
    "and view the predicted yoga pose."
)

selected_model_name = st.selectbox(
    "Select a model for inference:",
    list(MODEL_OPTIONS.keys())
)

uploaded_file = st.file_uploader(
    "Upload an image file",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    uploaded_image = Image.open(uploaded_file)

    st.subheader("Uploaded Image")
    st.image(uploaded_image, caption="Input image", use_container_width=True)

    if st.button("Run Prediction"):
        with st.spinner("Running inference..."):
            model_path = str(MODEL_OPTIONS[selected_model_name])
            interpreter = load_tflite_interpreter(model_path)

            input_data = preprocess_image(uploaded_image)
            probabilities = run_tflite_inference(interpreter, input_data)

            predicted_index = int(np.argmax(probabilities))
            predicted_class = CLASS_NAMES[predicted_index]
            predicted_confidence = float(probabilities[predicted_index])

            top_predictions = get_top_k_predictions(probabilities, CLASS_NAMES, k=3)

        st.success("Prediction complete.")

        st.subheader("Predicted Class")
        st.write(f"**Model:** {selected_model_name}")
        st.write(f"**Predicted yoga pose:** {predicted_class}")
        st.write(f"**Confidence:** {predicted_confidence:.4f}")

        st.subheader("Top 3 Predictions")
        top3_dict = {
            label: score for label, score in top_predictions
        }
        st.bar_chart(top3_dict)

        st.write("Top predictions:")
        for rank, (label, score) in enumerate(top_predictions, start=1):
            st.write(f"{rank}. {label} — {score:.4f}")
else:
    st.info("Please upload a JPG or PNG image to begin.")