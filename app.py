import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# ==========================================
# 1. PAGE SETUP & VIBRANT STYLING (CUSTOM CSS)
# ==========================================
st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="centered")

# Custom CSS for bubbly, gradient UI with a heartbeat animation button
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
""", unsafe_scale=True, unsafe_allow_html=True)

# ==========================================
# 2. MODEL TRAINING PREPARATION (YOUR EXACT CODE)
# ==========================================
@st.cache_resource
def load_and_train_model():
    # Load dataset
    df = pd.read_csv("Heart_disease_statlog.csv")
    
    col_features = ['age', 'sex', 'cp', 'thal', 'trestbps', 'chol', 'slope', 'ca', 'oldpeak', 'fbs', 'restecg', 'thalach', 'exang']
    X = df[col_features]
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Matching option 1 exact configuration
    std_scale_pipeline = Pipeline(steps=[('std_scaler', StandardScaler())])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("Scale_All_Numeric", std_scale_pipeline, 
             ['age', 'thalach', 'chol', 'oldpeak', 'trestbps', 'cp', 'thal', 'slope', 'ca'])
        ],
        remainder='passthrough'
    )
    
    final_pipeline = Pipeline(steps=[
        ('Std Scaling ', preprocessor),
        ('ml_model', LogisticRegression())
    ])
    
    final_pipeline.fit(X_train, y_train)
    return final_pipeline

# Initialize pipeline
model_pipeline = load_and_train_model()

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
    exang = st.selectbox("Exercise Induced Angina (exang)", options=[0, 1], format_func=lambda x: "Yes (1)"
  
