
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Lung Cancer Classification",
    page_icon="🫁",
    layout="centered"
)

MODEL_PATH = "best_cnn_model.keras"

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

def predict_lung_cancer(image):

    image = image.convert("RGB")
    image = image.resize((224, 224))

    image_array = np.array(image).astype("float32")
    image_array = image_array / 255.0

    image_batch = np.expand_dims(image_array, axis=0)

    probability = float(
        model.predict(image_batch, verbose=0)[0][0]
    )

    if probability >= 0.5:
        predicted_class = "Cancer"
    else:
        predicted_class = "Normal"

    return predicted_class, probability


st.title("🫁 Lung Cancer Classification")

st.write(
    "Upload a lung CT image to classify it as Normal or Cancer."
)

st.warning(
    "This application is an academic/research prototype "
    "and is not a clinical diagnostic system."
)

uploaded_file = st.file_uploader(
    "Upload Lung CT Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded CT Image",
        use_container_width=True
    )

    if st.button("Classify Image"):

        predicted_class, probability = predict_lung_cancer(image)

        normal_probability = 1 - probability

        st.subheader("Prediction")

        if predicted_class == "Cancer":
            st.error("Prediction: Cancer")
        else:
            st.success("Prediction: Normal")

        st.write(
            f"Normal score: {normal_probability:.4f}"
        )

        st.write(
            f"Cancer score: {probability:.4f}"
        )

        st.caption(
            "The displayed scores are model outputs, "
            "not clinical probabilities."
        )
