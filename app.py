import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, UnidentifiedImageError
from pathlib import Path

# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="CARCINOMA: An Intelligent AI Based Cancer Detection Model",
    page_icon="🫁",
    layout="wide",
)

# =========================================================
# MODEL CONFIGURATION
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_cnn_model.keras"
IMG_SIZE = (224, 224)

# =========================================================
# LOAD MODEL
# =========================================================
@st.cache_resource
def load_model(model_path: str):
    return tf.keras.models.load_model(model_path)

if not MODEL_PATH.is_file():
    st.error(
        f"Model file not found: {MODEL_PATH.name}. "
        "Make sure best_cnn_model.keras is in the same GitHub repository "
        "directory as app.py."
    )
    st.stop()

try:
    model = load_model(str(MODEL_PATH))
except Exception as exc:
    st.error("The CNN model could not be loaded. Check the model file and dependencies.")
    st.exception(exc)
    st.stop()

# =========================================================
# PREDICTION FUNCTION
# The trained model uses one sigmoid output:
# 0 = Normal, 1 = Cancer (threshold 0.5).
# =========================================================
def predict_lung_cancer(image: Image.Image):
    image = image.convert("RGB").resize(IMG_SIZE)
    image_array = np.asarray(image, dtype=np.float32) / 255.0
    image_batch = np.expand_dims(image_array, axis=0)

    output = np.asarray(model.predict(image_batch, verbose=0)).reshape(-1)
    if output.size != 1:
        raise ValueError(
            "Expected a single sigmoid output from the CNN model, "
            f"but received output shape {np.asarray(model.predict(image_batch, verbose=0)).shape}."
        )

    cancer_score = float(np.clip(output[0], 0.0, 1.0))
    predicted_class = "Cancer" if cancer_score >= 0.5 else "Normal"
    return predicted_class, cancer_score, 1.0 - cancer_score

# =========================================================
# HEADER
# =========================================================
st.markdown(
    """
    <h1 style="text-align:center; margin-bottom:5px;">🫁 CARCINOMA</h1>
    <h3 style="text-align:center; font-weight:400;">
        An Intelligent AI Based Cancer Detection Model
    </h3>
    <p style="text-align:center; color:gray;">
        Deep Learning Based Lung CT Image Classification
    </p>
    """,
    unsafe_allow_html=True,
)
st.divider()

# =========================================================
# PROJECT INFORMATION
# =========================================================
with st.expander("ℹ️ About CARCINOMA"):
    st.markdown(
        """
        ### CARCINOMA – Intelligent AI Based Cancer Detection Model

        This academic project uses a **Custom Convolutional Neural Network (CNN)**
        to classify uploaded lung CT images into two categories:

        - 🟢 **Normal**
        - 🔴 **Cancer**

        **Dataset:** IQ-OTH/NCCD Lung Cancer Dataset  
        **Model:** Custom Convolutional Neural Network  
        **Input size:** 224 × 224 pixels  

        **Reported test accuracy:** 88.48%  
        **Reported ROC-AUC:** 0.9560  
        **Reported cancer recall:** 94%  
        **Reported cancer F1-score:** 0.91  

        These metrics describe evaluation on the project's test dataset and do not
        guarantee the same performance on new images.
        """
    )

# =========================================================
# IMAGE UPLOAD
# =========================================================
st.subheader("📤 Upload Lung CT Image")
uploaded_file = st.file_uploader(
    "Choose a JPG, JPEG, or PNG image",
    type=["jpg", "jpeg", "png"],
    help="Upload a lung CT image for AI-based classification.",
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        image.load()
    except (UnidentifiedImageError, OSError, ValueError):
        st.error("The uploaded file could not be read as an image. Please upload a valid JPG, JPEG, or PNG file.")
        st.stop()

    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.subheader("🖼️ Uploaded CT Image")
        st.image(image, caption="Uploaded Lung CT Image", use_container_width=True)

    with col2:
        st.subheader("📋 Image Information")
        st.write(f"**Format:** {image.format or 'Unknown'}")
        st.write(f"**Original size:** {image.size[0]} × {image.size[1]} pixels")
        st.write("**Model input:** 224 × 224 pixels")
        st.write("**Color format:** RGB")
        st.info(
            "The image is converted to RGB, resized to 224 × 224 pixels, "
            "and normalized before being passed to the trained CNN."
        )

    st.divider()

    if st.button("🔍 Classify Image", type="primary", use_container_width=True):
        try:
            with st.spinner("Analyzing lung CT image..."):
                predicted_class, cancer_score, normal_score = predict_lung_cancer(image)
        except Exception as exc:
            st.error("Prediction failed. Please check the model output and try again.")
            st.exception(exc)
            st.stop()

        st.divider()

        if predicted_class == "Cancer":
            st.error("🔴 Cancer-Class Prediction")
            st.subheader("📄 AI Classification Report")
            st.markdown(
                """
                The uploaded lung CT image has been classified under the
                **Cancer** category by the trained **CARCINOMA** model.

               
            )

            st.subheader("⚠️ Possible Risk Factors Associated with Lung Cancer")
            st.markdown(
                """
                Lung cancer may be associated with several risk factors, including:

                **Tobacco smoking**  
                Smoking is a major risk factor for lung cancer.

                **Second-hand smoke exposure**  
                Exposure to tobacco smoke from others can increase risk.

                **Occupational exposure**  
                Exposure to asbestos, silica, arsenic, chromium, and certain industrial chemicals may increase risk.

                **Air pollution**  
                Long-term exposure to polluted air and fine particulate matter may contribute to risk.

                **Radon exposure**  
                Radon is a naturally occurring radioactive gas that can accumulate indoors.

                **Family history and genetic factors**  
                Family history may be associated with increased susceptibility.

                **Previous radiation exposure**  
                Certain previous radiation treatments involving the chest may be a risk factor.

                
                """
            )

            st.subheader("🏥 Medical Recommendation")
            st.info(
                "Please consult a qualified healthcare professional for an appropriate "
                "clinical evaluation. Do not make medical decisions based only on this AI result."
            )

        else:
            st.success("🟢 Normal-Class Prediction")
            st.subheader("📄 AI Classification Report")
            st.markdown(
        
                The uploaded lung CT image has been classified under the
                **Normal** category by the trained **CARCINOMA** model.

               
            )

            st.subheader("🫁 Healthy Lung Recommendations")
            st.markdown(
                """
                Although the model classified this image as Normal, maintaining good lung health is important.

                **Avoid tobacco smoking**  
                Avoid smoking and other forms of tobacco use.

                **Avoid second-hand smoke**  
                Reduce exposure to tobacco smoke from others whenever possible.

                **Reduce air-pollution exposure**  
                Reduce exposure to air pollution, dust, and harmful chemical fumes.

                **Use protective equipment**  
                Use appropriate protection when working in dusty or chemically exposed environments.

                **Maintain regular physical activity**  
                Follow a healthy lifestyle and stay physically active as appropriate for you.

                **Follow recommended medical check-ups**  
                Seek medical advice about check-ups based on your health history and risk factors.

                **Pay attention to persistent symptoms**  
                Seek medical attention for persistent cough, chest pain, breathing difficulty,
                unexplained weight loss, or coughing up blood.
                """
            )

            st.subheader("🏥 Medical Recommendation")
            st.info(
                "If you have persistent symptoms or medical concerns, consult a qualified "
                "healthcare professional even if the model classifies the image as Normal."
            )

        # Contact information supplied for this project.
        st.divider()
        st.subheader("📞 Medical Contact Information")
        st.markdown(
            """
            **Dr. Kavin Kumar, MBBS, MD (Oncology)**  
            Vivekanandha Medical College and Hospitals  
            **Contact:** 9443734417  

            **For prior report analysis:**  
            svmchriadmissions@vivekanandha.ac.in
            """
        )

# =========================================================
# FOOTER
# =========================================================
st.divider()
st.markdown(
    <div style="text-align:center; color:gray;">
        <b>CARCINOMA</b><br>
        An Intelligent AI Based Cancer Detection Model<br><br>
        VIVEKANANDHA COLLEGE OF ENGINEERING FOR WOMEN<br><br>
        Department of Computer Science and Engineering<br><br>
        <small>Academic / Research Project</small>
    </div>
    unsafe_allow_html=True,
)

