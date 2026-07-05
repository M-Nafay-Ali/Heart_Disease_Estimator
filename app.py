import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==========================================
# 1. PAGE SETUP & VIBRANT STYLING (CUSTOM CSS)
# ==========================================
st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="centered")

# Custom CSS for a bubbly, gradient UI with a heartbeat animation button
st.markdown("""
    <style>
    /* Main background gradient */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4e8f0 100%);
    }
    
    /* Bubbly white container cards */
    div[data-testid="stVerticalBlock"] > div {
        background-color: white;
        border-radius: 20px;
        padding: 20px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.05);
        margin-bottom: 15px;
    }
    
    /* Titles and text custom color */
    h1, h2, h3 {
        color: #2c3e50 !important;
        font-family: 'Poppins', sans-serif;
    }
    
    /* Heartbeat keyframes animation */
    @keyframes heartbeat {
        0% { transform: scale(1); }
        20% { transform: scale(1.1); }
        40% { transform: scale(1.05); }
        60% { transform: scale(1.15); }
        80% { transform: scale(1); }
        100% { transform: scale(1); }
    }
    
    /* Custom Styling for our Heart Predict Button */
    .stButton>button {
        background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%) !important;
        color: white !important;
        border: none !important;
        padding: 20px 40px !important;
        font-size: 24px !important;
        font-weight: bold !important;
        border-radius: 50px !important;
        box-shadow: 0 10px 20px rgba(255, 75, 43, 0.3) !important;
        transition: all 0.3s ease !important;
        display: block;
        margin: 0 auto !important;
        animation: heartbeat 2s infinite ease-in-out;
    }
    
    .stButton>button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 15px 25px rgba(255, 75, 43, 0.5) !important;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. LOAD THE SAVED KAGGLE PIPELINE
# ==========================================
@st.cache_resource
def load_saved_model():
    # This automatically loads your exact 92.59% accuracy pipeline
    return joblib.load("heart_model.pkl")

model_pipeline = load_saved_model()

# ==========================================
# 3. INTERACTIVE UI ELEMENTS
# ==========================================
st.title("❤️ CardioPulse Analyser")
st.write("Input patient metrics below to check cardiac health using our 92.59% accuracy pipeline.")

st.header("📋 Patient Clinical Vitals")

# Bubbly Columns layout for clean input controls
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age of Patient", 10, 100, 54)
    sex = st.selectbox("Biological Sex", options=[1, 0], format_func=lambda x: "Male (1)" if x == 1 else "Female (0)")
    cp = st.selectbox("Chest Pain Type (cp)", options=[0, 1, 2, 3], format_func=lambda x: f"Type {x}")
    trestbps = st.slider("Resting Blood Pressure (trestbps) mm Hg", 80, 200, 130)
    chol = st.slider("Serum Cholesterol (chol) mg/dl", 100, 600, 240)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", options=[0, 1], format_func=lambda x: "True (1)" if x == 1 else "False (0)")

with col2:
    restecg = st.selectbox("Resting ECG Results (restecg)", options=[0, 1, 2])
    thalach = st.slider("Max Heart Rate Achieved (thalach)", 60, 220, 150)
    exang = st.selectbox("Exercise Induced Angina (exang)", options=[0, 1], format_func=lambda x: "Yes (1)" if x == 1 else "No (0)")
    oldpeak = st.slider("ST Depression (oldpeak)", 0.0, 6.0, 1.0, step=0.1)
    slope = st.selectbox("Slope of Peak Exercise ST (slope)", options=[0, 1, 2])
    ca = st.selectbox("Major Vessels Colored by Fluoroscopy (ca)", options=[0, 1, 2, 3])
    thal = st.selectbox("Thalassemia Type (thal)", options=[1, 2, 3], format_func=lambda x: f"Type {x}")

# Arrange data precisely in the feature row order your pipeline expects
input_data = pd.DataFrame([{
    'age': age, 'sex': sex, 'cp': cp, 'thal': thal, 'trestbps': trestbps, 
    'chol': chol, 'slope': slope, 'ca': ca, 'oldpeak': oldpeak, 
    'fbs': fbs, 'restecg': restecg, 'thalach': thalach, 'exang': exang
}])

st.markdown("<br>", unsafe_allow_html=True)

# ==========================================
# 4. PREDICTION RUNNER & DYNAMIC OUTCOME
# ==========================================
# The Animated Beat Button!
if st.button("💓 ANALYZE HEART"):
    prediction = model_pipeline.predict(input_data)[0]
    probabilities = model_pipeline.predict_proba(input_data)[0]
    confidence = probabilities[prediction] * 100
    
    st.markdown("---")
    
    if prediction == 0:
        # Healthy Result Layout (Vibrant Teal Green)
        st.markdown(f"""
            <div style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); 
                        padding: 30px; border-radius: 20px; color: white; text-align: center;
                        box-shadow: 0 10px 20px rgba(56, 239, 125, 0.3);">
                <h2 style="color: white !important; margin: 0;">💚 Status: Normal Heart Activity</h2>
                <p style="font-size: 18px; margin-top: 10px;">The model predicts <strong>No Heart Disease Risk</strong> with a confidence score of <strong>{confidence:.1f}%</strong>.</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        # Risk Detected Layout (Vibrant Soft Red/Coral)
        st.markdown(f"""
            <div style="background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%); 
                        padding: 30px; border-radius: 20px; color: white; text-align: center;
                        box-shadow: 0 10px 20px rgba(255, 75, 43, 0.3);">
                <h2 style="color: white !important; margin: 0;">⚠️ Status: Cardiovascular Risk Detected</h2>
                <p style="font-size: 18px; margin-top: 10px;">The model predicts a <strong>High Risk of Heart Disease</strong> with a confidence score of <strong>{confidence:.1f}%</strong>.</p>
            </div>
        """, unsafe_allow_html=True)
                    
