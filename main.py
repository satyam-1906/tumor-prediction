import streamlit as st
import pandas as pd
import pickle

st.set_page_config(
    page_icon="🥼",
    page_title="Thyroid Cancer Risk Prediction",
    layout="wide"
    )
st.title("Thyroid Cancer Risk Prediction")
st.write("Enter the following details to predict the risk of thyroid cancer:")
age = st.number_input("Age", min_value=0, max_value=120, value=30)
gender = st.selectbox("Gender", options=["Male", "Female"])
country = st.selectbox("Country", options=["Brazil", "USA", "India", "China", "Germany", "Japan", "Nigeria", "South Korea", "UK", "Russia"])
ethnicity = st.selectbox("Ethnicity", options=["Asian", "Hispanic", 'Caucasian', "African", "Middle Eastern"])
fam_history = st.selectbox("Family History of Thyroid Cancer", options=["Yes", "No"])   
radiation_exp = st.selectbox("Radiation Exposure to Head/Neck", options=["Yes", "No"])  
iodine_def = st.selectbox("Iodine Deficiency", options=["Yes", "No"])   
smoking_status = st.selectbox("Smoking Status", options=["Yes", "No"])   
obesity = st.selectbox("Obesity", options=["Yes", "No"])
diabetes = st.selectbox("Diabetes", options=["Yes", "No"])
thyroid_risk = st.selectbox("Pre-existing Thyroid Conditions", options=["High", "Medium", "Low"])
tsh = st.number_input("TSH Level", min_value=0.0, value=2.0)
t4 = st.number_input("T4 Level", min_value=0.0, value=100.0)
t3 = st.number_input("T3 Level", min_value=0.0, value=1.0)
ns = st.number_input("Nodule Size (cm)", min_value=0.0, value=1.0)  
features = pd.DataFrame({
    'Age': [age],
    'TSH_Level': [tsh],
    'T3_Level': [t3],
    'T4_Level': [t4],
    'Nodule_Size':[ns],
    'Gender_Female': [1 if gender == "Female" else 0],
    'Gender_Male': [1 if gender == "Male" else 0],
    'Country_Brazil': [1 if country == "Brazil" else 0],
    'Country_China': [1 if country == "China" else 0],
    'Country_Germany': [1 if country == "Germany" else 0],
    'Country_India': [1 if country == "India" else 0],
    'Country_Japan': [1 if country == "Japan" else 0],
    'Country_Nigeria': [1 if country == "Nigeria" else 0],
    'Country_Russia': [1 if country == "Russia" else 0],
    'Country_South Korea': [1 if country == "South Korea" else 0],
    'Country_UK': [1 if country == "UK" else 0],
    'Country_USA': [1 if country == "USA" else 0],
    'Ethnicity_African': [1 if ethnicity == "African" else 0],
    'Ethnicity_Asian': [1 if ethnicity == "Asian" else 0],
    'Ethnicity_Caucasian': [1 if ethnicity == "Caucasian" else 0],
    'Ethnicity_Hispanic': [1 if ethnicity == "Hispanic" else 0],
    'Ethnicity_Middle Eastern': [1 if ethnicity == "Middle Eastern" else 0],
    'Family_History_No': [1 if fam_history == "No" else 0],
    'Family_History_Yes': [1 if fam_history == "Yes" else 0],
    'Radiation_Exposure_No': [1 if radiation_exp == "No" else 0],
    'Radiation_Exposure_Yes': [1 if radiation_exp == "Yes" else 0],
    'Iodine_Deficiency_No': [1 if iodine_def == "No" else 0],
    'Iodine_Deficiency_Yes': [1 if iodine_def == "Yes" else 0],
    'Smoking_No': [1 if smoking_status == "No" else 0],
    'Smoking_Yes': [1 if smoking_status == "Yes" else 0],
    'Obesity_No': [1 if obesity == "No" else 0],
    'Obesity_Yes': [1 if obesity == "Yes" else 0],
    'Diabetes_No': [1 if diabetes == "No" else 0],
    'Diabetes_Yes': [1 if diabetes == "Yes" else 0],
    'Thyroid_Cancer_Risk_High': [1 if thyroid_risk == "High" else 0],
    'Thyroid_Cancer_Risk_Low': [1 if thyroid_risk == "Low" else 0],
    'Thyroid_Cancer_Risk_Medium': [1 if thyroid_risk == "Medium" else 0],
    'Indicator_1': [tsh/t3 if t3 != 0 else 0],
    'Indicator_2': [tsh/t4 if t4 != 0 else 0],
    'Indicator_3': [age/tsh if tsh != 0 else 0]
})
if st.button("Predict Risk"):
    with open('thyroid_cancer_risk_model.pkl', 'rb') as f:
        model = pickle.load(f)
    risk = model.predict(features)[0]
    if risk == "Malignant":
        st.write("Malignant")
    else:
        st.write("Benign")
    risk_prob = model.predict_proba(features)[0][1]*100
    st.write(f"Prediction Confidence: {risk_prob:.2f}")
    if risk == "Malignant":
        st.error("High risk of thyroid cancer detected. Please consult a healthcare professional.")
    else:
        st.success("Low risk of thyroid cancer detected.")

st.write("Disclaimer: This tool is for informational purposes only and is not a substitute for professional medical advice.")   