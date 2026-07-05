# Heart_Disease_Estimator
# Note all the projects till now are made in mobile so pls be patient with my mistakes
# 🫀 Cardiovascular Risk Assessment & Predictive Modeling

An interactive, machine learning web application that predicts the risk of heart disease based on clinical patient vitals. This project achieves an evaluation accuracy of **92.59%** using a clean, production-ready Scikit-Learn data pipeline.

🌍 **Live Web App Link:** [https://heartdiseaseestimator.streamlit.app/](https://heartdiseaseestimator.streamlit.app/)

---

## 🚀 Project Overview

The goal of this project is to take clinical data and build an accurate, fast predictor for cardiovascular health. Instead of manually preprocessing training and testing data separately, this project builds a robust **Data Pipeline** that automates scaling and processing, completely eliminating data leakage risks.

### ✨ Key Features
* **Machine Learning Pipeline:** Uses `ColumnTransformer` and `StandardScaler` paired with a tuned `LogisticRegression` classifier.
* **Lively Web Interface:** Built using Streamlit, featuring custom bubbly UI cards, smooth background gradients, and an animated, glowing heartbeat action button.
* **Dynamic Feedback:** The dashboard changes color based on the risk category—soothing teal for normal health and vibrant red for high-risk flags.

---

## 📊 Dataset Profile & Key Insights

The model is trained using the **Statlog Heart Disease Dataset** containing 270 perfectly clean rows and 14 clinical parameters. 

### Exploratory Data Insights:
* **Top Positive Predictors:** Features like Thalassemia Type (`thal`), Number of Major Vessels (`ca`), and Chest Pain Type (`cp`) show a strong positive correlation with heart disease risk.
* **The Main Negative Driver:** Maximum Heart Rate Achieved (`thalach`) shows a strong negative relationship. As maximum heart rate levels drop, the likelihood of an active heart risk classification increases.

---

## 📈 Model Performance & Evaluation

The final model pipeline was validated using a hidden 20% test partition (54 rows total).

### 1. Classification Metrics
* **Overall Accuracy:** 92.59%
* **Precision (Class 1 - Disease):** 95% (Extremely low false alarm rate)
* **Recall (Class 1 - Disease):** 86% (Successfully catches the vast majority of true risk cases)

### 2. Confusion Matrix Breakdown
* **True Negatives:** 32 healthy patients correctly categorized.
* **True Positives:** 18 risk cases correctly flagged.
* **False Positives:** Only 1 healthy patient safely over-predicted.
* **False Negatives:** Only 3 risk cases missed out of the entire partition.

---

## 📁 Repository Map

* `app.py` - The Streamlit interactive application script.
* `requirements.txt` - Python dependencies needed to build the hosting container (`scikit-learn==1.6.1`).
* `heart_model.pkl` - The trained, exported pipeline model file.
* `Heart_disease_statlog.csv` - The original source data file.

---

## 🛠️ How to Launch Locally

If you want to pull down this code and run the app on your local machine:

1. Clone this repository to your local directory.
2. Install the exact package dependencies:
   ```bash
   pip install -r requirements.txt

## 🤝 Contact Information:-

***Github:-*** [https://github.com/M-Nafay-Ali]

***LinkedIn:-*** [https://www.linkedin.com/in/mohammed-nafay-ali-16519138a?utm_source=share_via&utm_content=profile&utm_medium=member_android]

***Email:-*** [englandengland271@gmail.com]

