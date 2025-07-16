import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
# Load the model
model = joblib.load('xgboost_model.pkl')
# Define feature options
Diagnose_options = {    1: "特发", 2: '不典型',    3: '继发' ,  4: '抗体诊断' , 5:"未知"}
# Define feature names
feature_names = [   "总胆汁酸TBA（μmol/L）", "前白蛋白PA（mg/L）",  "红细胞（尿液）（个/HPF）", "二氧化碳", "甘胆酸CG（mg/L）", "红细胞体积分布宽度CVRDW（%）",    "淋巴细胞绝对值Lymph#（×10⁹/L）",  "嗜碱性粒细胞绝对值Baso#（×10⁹/L）",
                "碱性磷酸酶ALP（IU/L）", "总胆固醇TC（mmol/L）", "嗜酸性粒细胞绝对值Baso#（×10⁹/L）","平均血小板体积MPV（fl）","具体诊断"，"β2微球蛋白β2-MG（mg/L）"，"低密度脂蛋白胆固醇LDL-C（mmol/L）"]
# Streamlit user interface
st.title("麻附益肾方治疗膜性肾病疗效预测")


# 设置
TBA = st.number_input("总胆汁酸TBA（μmol/L）:", min_value=0.0, max_value=20.0, value=3.0,format="%.1f")

PA = st.number_input("前白蛋白PA（mg/L）:", min_value=0.0, max_value=600.0, value=300.0,format="%.1f")

RBC_U = st.number_input("红细胞（尿液）（个/HPF）:", min_value=0.00, max_value=100.00, value=10.00,format="%.2f")

CO2 = st.number_input("二氧化碳:", min_value=0.0, max_value=50.0, value=25.0,format="%.1f")

CG = st.number_input("甘胆酸CG（mg/L）:", min_value=0.00, max_value=5.00, value=1.00,format="%.2f")

CVRDW = st.number_input("红细胞体积分布宽度CVRDW（%）:", min_value=0.0, max_value=20.0, value=12.0,format="%.1f")

"Lymph#" =st.number_input("淋巴细胞绝对值Lymph#（×10⁹/L）:", min_value=0.00, max_value=50.00, value=25.00,format="%.2f")

"Baso#" =st.number_input("嗜碱性粒细胞绝对值Baso#（×10⁹/L）:", min_value=0.00, max_value=1.00, value=0.10,format="%.2f")

ALP = st.number_input("碱性磷酸酶ALP（IU/L）:", min_value=0.0, max_value=1000.0, value=300.0,format="%.1f")

TC = st.number_input("总胆固醇TC（mmol/L）:", min_value=0.00, max_value=20.00, value=4.00,format="%.2f")

"Eos#" = st.number_input("嗜酸性粒细胞绝对值Baso#（×10⁹/L）:", min_value=0.00, max_value=1.00, value=0.10,format="%.2f")

MPV = st.number_input("平均血小板体积MPV（fl）:", min_value=0.0, max_value=20.0, value=5.0,format="%.1f")

具体诊断 = st.selectbox("具体诊断:", options=list(Diagnose_options.keys()), format_func=lambda x: Diagnose_options[x])

β2_MG = st.number_input("β2微球蛋白β2-MG（mg/L）:", min_value=0.00, max_value=20.00, value=1.00,format="%.2f")

LDL_C = st.number_input("低密度脂蛋白胆固醇LDL-C（mmol/L）:", min_value=0.00, max_value=20.00, value=3.00,format="%.2f")






# Process inputs and make predictions
feature_values = [TBA, PA,  RBC_U, CO2, CG, CVRDW, "Lymph#", "Baso#",ALP, TC, "Eos#",MPV, 具体诊断,β2_MG,LDL_C]
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