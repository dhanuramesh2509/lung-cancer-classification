import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="CARCINOMA: An Intelligent AI Based Cancer Detection Model",
    page_icon="🫁",
    layout="wide"
)

# =========================================================
# MODEL CONFIGURATION
# =========================================================

MODEL_PATH = "best_cnn_model.keras"
IMG_SIZE = (224, 224)

# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


model = load_model()

# =========================================================
# PREDICTION FUNCTION
# =========================================================

def predict_lung_cancer(image):

    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)

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

    return predicted_class


# =========================================================
# HEADER
# =========================================================

st.markdown(
    """
    <h1 style="text-align:center; margin-bottom:5px;">
        🫁 CARCINOMA
    </h1>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <h3 style="text-align:center; font-weight:400;">
        An Intelligent AI Based Cancer Detection Model
    </h3>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <p style="text-align:center; color:gray;">
        Deep Learning Based Lung CT Image Classification
    </p>
    """,
    unsafe_allow_html=True
)

st.divider()


# =========================================================
# MEDICAL DISCLAIMER
# =========================================================

st.warning(
    """
    ⚠️ **IMPORTANT MEDICAL DISCLAIMER**

    CARCINOMA is an academic/research prototype developed for
    lung CT image classification.

    The AI-generated result is only a model prediction and does
    not constitute a confirmed medical diagnosis. It should not
    be used as a substitute for professional medical evaluation.
    """
)


# =========================================================
# PROJECT INFORMATION
# =========================================================

with st.expander("ℹ️ About CARCINOMA"):

    st.markdown(
        """
        ### CARCINOMA – Intelligent AI Based Cancer Detection Model

        This project uses a **Custom Convolutional Neural Network
        (CNN)** to classify lung CT images into two categories:

        - 🟢 **Normal**
        - 🔴 **Cancer**

        ### Dataset

        **IQ-OTH/NCCD Lung Cancer Dataset**

        ### Model

        **Custom Convolutional Neural Network**

        ### Input Image Size

        **224 × 224 pixels**

        ### Model Performance

        **Test Accuracy:** 88.48%

        **ROC-AUC:** 0.9560

        **Cancer Recall:** 94%

        **Cancer F1-Score:** 0.91

        The system is intended for academic and research purposes.
        """
    )


# =========================================================
# IMAGE UPLOAD
# =========================================================

st.subheader("📤 Upload Lung CT Image")

uploaded_file = st.file_uploader(
    "Choose a JPG, JPEG, or PNG image",
    type=["jpg", "jpeg", "png"],
    help="Upload a lung CT image for AI-based classification."
)


# =========================================================
# IMAGE DISPLAY
# =========================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    col1, col2 = st.columns([1.2, 1])

    # -----------------------------------------------------
    # IMAGE
    # -----------------------------------------------------

    with col1:

        st.subheader("🖼️ Uploaded CT Image")

        st.image(
            image,
            caption="Uploaded Lung CT Image",
            use_container_width=True
        )

    # -----------------------------------------------------
    # IMAGE INFORMATION
    # -----------------------------------------------------

    with col2:

        st.subheader("📋 Image Information")

        st.write(
            f"**Format:** {image.format}"
        )

        st.write(
            f"**Original Size:** "
            f"{image.size[0]} × {image.size[1]}"
        )

        st.write(
            "**Model Input:** 224 × 224 pixels"
        )

        st.write(
            "**Color Format:** RGB"
        )

        st.info(
            """
            The uploaded image is resized to 224 × 224 pixels
            and normalized before being passed to the trained
            CNN model.
            """
        )

    st.divider()


    # =====================================================
    # CLASSIFICATION BUTTON
    # =====================================================

    if st.button(
        "🔍 Classify Image",
        type="primary",
        use_container_width=True
    ):

        with st.spinner(
            "Analyzing lung CT image..."
        ):

            predicted_class = predict_lung_cancer(image)


        st.divider()


        # =================================================
        # CANCER RESULT
        # =================================================

        if predicted_class == "Cancer":

            st.error(
                "🔴 Cancer-Class Prediction"
            )

            st.subheader(
                "📄 AI Classification Report"
            )

            st.markdown(
                """
                The uploaded lung CT image has been classified
                under the **Cancer category** by the trained
                **CARCINOMA Model**.

                This is an AI-based image classification result
                and **does not confirm the presence of cancer**.
                Professional medical evaluation is required for
                diagnosis.
                """
            )


            # -------------------------------------------------
            # RISK FACTORS
            # -------------------------------------------------

            st.subheader(
                "⚠️ Possible Risk Factors Associated with Lung Cancer"
            )

            st.markdown(
                """
                Lung cancer may be associated with several risk
                factors, including:

                **• Tobacco smoking**

                A major risk factor for lung cancer.

                **• Second-hand smoke exposure**

                Long-term exposure to tobacco smoke from others.

                **• Occupational exposure**

                Exposure to asbestos, silica, arsenic, chromium,
                and certain industrial chemicals.

                **• Air pollution**

                Long-term exposure to polluted air and fine
                particulate matter.

                **• Radon exposure**

                Exposure to naturally occurring radioactive
                radon gas.

                **• Family history and genetic factors**

                A family history of lung cancer may increase
                susceptibility.

                **• Previous radiation exposure**

                Certain previous radiation treatments involving
                the chest.

                **• Age and environmental factors**

                Risk generally increases with age and prolonged
                environmental exposure.
                """
            )


            st.divider()


            # -------------------------------------------------
            # RECOMMENDATION
            # -------------------------------------------------

            st.subheader(
                "🏥 Immediate Recommendation"
            )

            st.info(
                """
                Please consult a qualified oncologist or other
                appropriate medical professional for a complete
                medical evaluation.

                The AI classification should not be used as a
                substitute for professional medical diagnosis.
                """
            )


            # -------------------------------------------------
            # DOCTOR INFORMATION
            # -------------------------------------------------

            st.markdown(
                """
                ### 👨‍⚕️ Medical Contact

                **Dr. Kavin Kumar, MBBS, MD (Oncology)**

                **Vivekanandha Medical College and Hospitals**

                **Contact:** 9443734417
                """
            )

            st.markdown(
                """
                **For prior report analysis:**

                svmchriadmissions@vivekanandha.ac.in
                """
            )


        # =================================================
        # NORMAL RESULT
        # =================================================

        else:

            st.success(
                "🟢 Normal-Class Prediction"
            )

            st.subheader(
                "📄 AI Classification Report"
            )

            st.markdown(
                """
                The uploaded lung CT image has been classified
                under the **Normal category** by the trained
                **CARCINOMA Model**.

                Based on the visual patterns analyzed by the AI
                model, the uploaded image has been classified
                under the Normal category.

                However, a Normal AI classification does not
                completely rule out disease or other medical
                conditions.
                """
            )


            # -------------------------------------------------
            # HEALTH RECOMMENDATIONS
            # -------------------------------------------------

            st.subheader(
                "🫁 Healthy Lung Recommendations"
            )

            st.markdown(
                """
                Although the current AI classification is Normal,
                maintaining good lung health is important.

                **• Avoid tobacco smoking**

                Avoid smoking and other forms of tobacco use.

                **• Avoid second-hand smoke**

                Reduce exposure to tobacco smoke from others
                whenever possible.

                **• Reduce air-pollution exposure**

                Reduce exposure to air pollution, dust, and
                harmful chemical fumes.

                **• Use protective equipment**

                Use appropriate protective equipment when working
                in dusty or chemically exposed environments.

                **• Maintain regular physical activity**

                Follow a healthy lifestyle and maintain regular
                physical activity.

                **• Follow recommended medical check-ups**

                Regular medical check-ups may be appropriate,
                particularly for people with respiratory symptoms
                or known risk factors.

                **• Pay attention to persistent symptoms**

                Seek medical attention if you experience
                persistent cough, chest pain, breathing difficulty,
                unexplained weight loss, or coughing up blood.
                """
            )


            st.divider()


            # -------------------------------------------------
            # MEDICAL RECOMMENDATION
            # -------------------------------------------------

            st.subheader(
                "🏥 Medical Recommendation"
            )

            st.info(
                """
                If you have persistent symptoms or other medical
                concerns, consult a qualified medical professional
                for appropriate evaluation.
                """
            )


            # -------------------------------------------------
            # DOCTOR INFORMATION
            # -------------------------------------------------

            st.markdown(
                """
                ### 👨‍⚕️ Medical Contact

                **Dr. Kavin Kumar, MBBS, MD (Oncology)**

                **Vivekanandha Medical College and Hospitals**

                **Contact:** 9443734417
                """
            )

            st.markdown(
                """
                **For prior report analysis:**

                svmchriadmissions@vivekanandha.ac.in
                """ 
            )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.markdown(
    """
    <div style="text-align:center; color:gray;">

    <b>CARCINOMA</b>

    <br>

    An Intelligent AI Based Cancer Detection Model

    <br><br>

    VIVEKANANDHA COLLEGE OF ENGINEERING FOR WOMEN

    <br><br>

    Department of Computer Science and Engineering

    <br><br>

    <small>
    Academic / Research Project
    </small>

    </div>
    """,
    unsafe_allow_html=True
)