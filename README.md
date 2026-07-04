# Heart_Disease_Estimator
# 🫀 Cardiovascular Risk Assessment & Predictive Modeling

An end-to-end Machine Learning classification project designed to predict heart attack probability based on patient clinical attributes, achieving a audited, leakage-free 91% accuracy baseline.

## 🛡️ Data Integrity & Leakage Audit
* **Zero Target Leakage:** Verified that no post-diagnostic clinical markers or proxy target variables were included in the training pipeline.
* **Validation Strategy:** Evaluated using Stratified K-Fold cross-validation to guarantee that class distributions (At-Risk vs. Healthy) remained consistent across splits, preventing optimistic bias.

## 📊 Feature Architecture
The pipeline processes key clinical vectors including:
* **Biometric Markers:** Age, Sex, Resting Blood Pressure (`trestbps`), and Serum Cholesterol (`chol`).
* **Physiological Indicators:** Maximum Heart Rate Achieved (`thalach`) and ST Depression (`oldpeak`).
* **Categorical Predictors:** Chest Pain Type (`cp`), Fasting Blood Sugar (`fbs`), and Resting Electrocardiographic Results (`restecg`).
