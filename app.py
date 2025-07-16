import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
# Load the model
model = joblib.load('xgboost_model.pkl')
# Define feature options
cp_options = {    1: 'Typical angina (1)',    2: 'Atypical angina (2)',    3: 'Non-anginal pain (3)',    4: 'Asymptomatic (4)'}
restecg_options = {    0: 'Normal (0)',    1: 'ST-T wave abnormality (1)',    2: 'Left ventricular hypertrophy (2)'}
slope_options = {    1: 'Upsloping (1)',    2: 'Flat (2)',    3: 'Downsloping (3)'}
thal_options = {    1: 'Normal (1)',    2: 'Fixed defect (2)',    3: 'Reversible defect (3)'}
# Define feature names
feature_names = [    "前白蛋白PA（mg/L）", "总胆汁酸TBA（μmol/L）", "红细胞（尿液）（个/HPF）", "甘胆酸CG（mg/L）", "红细胞体积分布宽度CVRDW（%）",    "二氧化碳", "淋巴细胞绝对值Lymph#（×10⁹/L）", "总胆固醇TC（mmol/L）", "嗜碱性粒细胞绝对值Baso#（×10⁹/L）",
                "低密度脂蛋白胆固醇LDL-C（mmol/L）", "红细胞计数RBC（×10¹²／L）", "碱性磷酸酶ALP（IU/L）", "平均血小板体积MPV（fl）","白细胞（尿液）（个/HPF）"]
# Streamlit user interface
st.title("麻附益肾方治疗膜性肾病疗效预测")

# 设置
PA = st.number_input("前白蛋白PA（mg/L）:", min_value=0, max_value=600, value=300,format="%.2f")

TBA = st.number_input("总胆汁酸TBA（μmol/L）:", min_value=0, max_value=20, value=3,format="%.2f")

RBC_U = st.number_input("红细胞（尿液）（个/HPF）:", min_value=0, max_value=100, value=10,format="%.2f")

CG = st.number_input("甘胆酸CG（mg/L）:", min_value=0, max_value=5, value=1,format="%.2f")

CVRDW = st.number_input("红细胞体积分布宽度CVRDW（%）:", min_value=0, max_value=20, value=12,format="%.2f")

CO2 = st.number_input("二氧化碳:", min_value=0, max_value=50, value=25,format="%.2f")

Lymph =st.number_input("淋巴细胞绝对值Lymph#（×10⁹/L）:", min_value=0, max_value=50, value=25,format="%.2f")

TC = st.number_input("总胆固醇TC（mmol/L）:", min_value=0, max_value=20, value=4,format="%.2f")

Baso =st.number_input("嗜碱性粒细胞绝对值Baso#（×10⁹/L）:", min_value=0, max_value=1, value=0.1,format="%.2f")

LDL_C = st.number_input("低密度脂蛋白胆固醇LDL-C（mmol/L）:", min_value=0, max_value=20, value=3,format="%.2f")

RBC = st.number_input("红细胞计数RBC（×10¹²／L）:", min_value=0, max_value=20, value=4,format="%.2f")

ALP = st.number_input("碱性磷酸酶ALP（IU/L）:", min_value=0, max_value=1000, value=300,format="%.2f")

MPV = st.number_input("平均血小板体积MPV（fl）:", min_value=0, max_value=20, value=5,format="%.2f")

WBC_U = st.number_input("白细胞（尿液）（个/HPF）:", min_value=0, max_value=100, value=10,format="%.2f")




# Process inputs and make predictions
feature_values = [PA, TBA, RBC_U, CG, CVRDW, CO2, Lymph, TC, Baso, LDL_C, RBC, ALP, MPV,WBC_U]
features = np.array([feature_values])
if st.button("Predict"):    
    # Predict class and probabilities
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]

    # Display prediction results
    st.write(f"**Predicted Class:** {predicted_class}")    
    st.write(f"**Prediction Probabilities:** {predicted_proba}")

    # Generate advice based on prediction results    
    
    probability = predicted_proba[predicted_class] * 100

    if predicted_class == 1:        
        advice = (            
            f"根据我们的模型，预测您服用麻附益肾方治疗膜性肾病有效 "            
            f"模型预测您治疗有效的概率为{probability:.1f}%. "                   
        )    
    else:        
        advice = (            
            f"根据我们的模型，预测您服用麻附益肾方治疗膜性肾病无效 "            
            f"模型预测您治疗无效的概率为{probability:.1f}%. "             
        )

    st.write(advice)
    # Calculate SHAP values and display force plot    
    explainer = shap.TreeExplainer(model)    
    shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_names))
    shap.force_plot(explainer.expected_value, shap_values[0], pd.DataFrame([feature_values], columns=feature_names), matplotlib=True)
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
    st.image("shap_force_plot.png")