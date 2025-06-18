import streamlit as st
from PIL import Image
from utils import predict, get_model_metrics  # Import from utils.py

# Custom Theme and Page Config
st.set_page_config(page_title="Arthonova KOA Predictor", layout="wide")

# Sidebar Styling
st.sidebar.image("assets/0_LOGO.png")
st.sidebar.title("🔹 Navigation")
page = st.sidebar.radio("", ["🏠 Home", "📸 Check KOA", "📊 Model Predictions", "📈 Model Charts"])

# 🔹 Home Page
if page == "🏠 Home":

    st.image("assets/1_bg.png", use_container_width=True) 

    st.image("assets/2_aboutus.png", use_container_width=True) 

    st.image("assets/3_objective.png", use_container_width=True) 
    
    #st.markdown('<br>', unsafe_allow_html=True)

    st.markdown('<p style="text-align: center; color: gray; font-size: 14px;">Made by Group 6 | Fardeen, Dhruvisha and Deep.</p>', unsafe_allow_html=True)

# 🔹 Upload & KOA Detection Page
elif page == "📸 Check KOA":
    st.title("📸 Upload Your X-ray for KOA Detection")
    
    uploaded_file = st.file_uploader("Choose an X-ray Image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="🦴 Uploaded X-ray", width=700)

        # Predict using the model
        label, confidence = predict(image)

        st.subheader(f"🩺 Diagnosis: **{label}**")
        st.write(f"🔍 Confidence: **{confidence:.2f}%**")

        # Diagnostic Findings
        st.markdown("### 🔹 Diagnostic Findings:")
        if label == "Healthy":
            st.info("✅ No significant abnormalities detected.")
        elif label == "Doubtful":
            st.info("⚠️ Possible early-stage KOA with minor cartilage wear.")
        elif label == "Mild":
            st.info("🔸 Mild KOA detected, with slight joint space narrowing.")
        elif label == "Moderate":
            st.info("🔶 Moderate KOA with noticeable joint damage and osteophyte formation.")
        elif label == "Severe":
            st.info("🛑 Severe KOA with significant cartilage loss and bone deformity.")

        # Suggested Next Steps
        st.markdown("### 💡 Suggested Next Steps:")
        if label == "Healthy":
            st.success("1️⃣ Maintain an active lifestyle.\n\n2️⃣ Follow a balanced diet for joint health.\n\n3️⃣ Monitor for any future symptoms.")
        elif label == "Doubtful":
            st.warning("1️⃣ Regularly monitor symptoms.\n\n2️⃣ Start light exercises to maintain mobility.\n\n3️⃣ Consult a doctor if discomfort increases.")
        elif label == "Mild":
            st.warning("1️⃣ Consider physical therapy for joint flexibility.\n\n2️⃣ Use mild pain relievers if needed.\n\n3️⃣ Maintain a healthy weight to reduce joint stress.")
        elif label == "Moderate":
            st.error("1️⃣ Consult an orthopedic specialist for treatment options.\n\n2️⃣ Consider lifestyle modifications and joint-support supplements.\n\n3️⃣ Use assistive devices if experiencing pain during movement.")
        elif label == "Severe":
            st.error("1️⃣ Seek immediate medical consultation for treatment options.\n\n2️⃣ Surgical intervention (such as knee replacement) may be necessary.\n\n3️⃣ Pain management and mobility support strategies should be prioritized.")

# 🔹 Model Predictions Page
elif page == "📊 Model Predictions":
    st.title("📊 Model Performance Metrics")

    # Fetch metrics from utils.py
    metrics = get_model_metrics()

    st.markdown('<br>', unsafe_allow_html=True)

    # Display key metrics in a structured layout
    st.markdown("### 🔹 Performance Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("⚡ Accuracy", f"{metrics['accuracy']:.2%}")
    col2.metric("🎯 Precision", f"{metrics['precision']:.2%}")
    col3.metric("🔄 Recall", f"{metrics['recall']:.2%}")
    col4.metric("📏 F1-Score", f"{metrics['f1_score']:.2%}")

    st.markdown('<br>', unsafe_allow_html=True)

    # Display Confusion Matrix with interpretation
    st.markdown("### 🖼️ Confusion Matrix")
    st.image("assets/conf-mat.png", caption="Confusion Matrix", width=650)
    st.info("""
    **Interpretation:**  
    - The model accurately classifies "Healthy" and "Doubtful" KOA grades with high confidence.  
    - Severe KOA cases are mostly predicted correctly, which is crucial for medical diagnosis.
    """)



# 🔹 Model Charts Page
elif page == "📈 Model Charts":
    st.title("📈 Model Insights & Analytics")

    st.markdown('<br>', unsafe_allow_html=True)

    # Training and Validation Accuracy
    st.markdown("### 📊 Training & Validation Accuracy")
    st.image("assets/acc.png", caption="Accuracy Curve", width=600)
    st.success("""
    **Interpretation:**  
    - Both accuracies show a consistent upward trend, indicating effective learning.  
    - The validation accuracy closely follows the training accuracy, showing good generalization.
    """)

    st.markdown('<br>', unsafe_allow_html=True)

    # ROC Curve
    st.markdown("### 🎯 ROC Curve")
    st.image("assets/roc.png", caption="Receiver Operating Characteristic (ROC) Curve", width=600)
    st.info("""
    **Interpretation:**  
    - High AUC values (≥ 0.90) across all classes confirm strong classification capability.  
    - The model performs exceptionally well in detecting "Moderate" and "Severe" KOA cases (AUC = 0.98, 0.99).
    """)

    st.markdown('<br>', unsafe_allow_html=True)

    # Training and Validation Loss
    st.markdown("### 📉 Training & Validation Loss")
    st.image("assets/loss.png", caption="Loss Curve", width=600)
    st.warning("""
    **Interpretation:**  
    - The steady decline in loss demonstrates good model convergence.  
    - The training process is effective, leading to improved performance over epochs.
    """)

    st.markdown('<br>', unsafe_allow_html=True)

    # Precision-Recall Curve
    st.markdown("### 🎯 Precision-Recall Curve")
    st.image("assets/precall.png", caption="Precision-Recall Curve", width=600)
    st.success("""
    **Interpretation:**  
    - The model maintains high precision and recall for critical KOA grades like "Healthy" and "Severe," ensuring reliable classification.  
    - The curves indicate strong performance in distinguishing relevant cases from irrelevant ones.
    """)

