import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import warnings

# 忽略警告
warnings.filterwarnings("ignore")

# Load the model
model = joblib.load('feature_info.pkl')

# Define feature options
Diagnose_options = {
    1: "特发", 
    2: '不典型',    
    3: '继发', 
    4: '抗体诊断',
    5: "未知"
}

# Define feature names (确保与模型训练时的特征顺序一致)
feature_names = ["TBA","PA","RBC_U","CO2","CG","CVRDW","Lymph_abs","Baso_abs","ALP","TC","Eos_abs" , "MPV" ,"Diagnose" ,"β2_MG" , "LDL_C"]

# Streamlit user interface
st.title("麻附益肾方治疗膜性肾病疗效预测")

# 输入字段
TBA = st.number_input("总胆汁酸TBA（μmol/L）:", min_value=0.0, max_value=20.0, value=3.0, format="%.1f")
PA = st.number_input("前白蛋白PA（mg/L）:", min_value=0.0, max_value=600.0, value=300.0, format="%.1f")
RBC_U = st.number_input("红细胞（尿液）（个/HPF）:", min_value=0.00, max_value=100.00, value=10.00, format="%.2f")
CO2 = st.number_input("二氧化碳:", min_value=0.0, max_value=50.0, value=25.0, format="%.1f")
CG = st.number_input("甘胆酸CG（mg/L）:", min_value=0.00, max_value=5.00, value=1.00, format="%.2f")
CVRDW = st.number_input("红细胞体积分布宽度CVRDW（%）:", min_value=0.0, max_value=20.0, value=12.0, format="%.1f")
Lymph_abs = st.number_input("淋巴细胞绝对值Lymph#（×10⁹/L）:", min_value=0.00, max_value=50.00, value=25.00, format="%.2f")
Baso_abs = st.number_input("嗜碱性粒细胞绝对值Baso#（×10⁹/L）:", min_value=0.00, max_value=1.00, value=0.10, format="%.2f")
ALP = st.number_input("碱性磷酸酶ALP（IU/L）:", min_value=0.0, max_value=1000.0, value=300.0, format="%.1f")
TC = st.number_input("总胆固醇TC（mmol/L）:", min_value=0.00, max_value=20.00, value=4.00, format="%.2f")
Eos_abs = st.number_input("嗜酸性粒细胞绝对值Baso#（×10⁹/L）:", min_value=0.00, max_value=1.00, value=0.10, format="%.2f")
MPV = st.number_input("平均血小板体积MPV（fl）:", min_value=0.0, max_value=20.0, value=5.0, format="%.1f")
Diagnose = st.selectbox("具体诊断:", options=list(Diagnose_options.keys()), format_func=lambda x: Diagnose_options[x])
β2_MG = st.number_input("β2微球蛋白β2-MG（mg/L）:", min_value=0.00, max_value=20.00, value=1.00, format="%.2f")
LDL_C = st.number_input("低密度脂蛋白胆固醇LDL-C（mmol/L）:", min_value=0.00, max_value=20.00, value=3.00, format="%.2f")

# 确保特征顺序正确
feature_values = [TBA, PA, RBC_U, CO2, CG, CVRDW, Lymph_abs, Baso_abs, ALP, TC, Eos_abs, MPV, Diagnose, β2_MG, LDL_C]
features = np.array([feature_values])

if st.button("Predict"):

        # 预测
        predicted_class = model.predict(features)[0]
        predicted_proba = model.predict_proba(features)[0]
        
        # 显示结果
        st.write(f"**Predicted Class:** {predicted_class}")
        st.write(f"**Prediction Probabilities:** {predicted_proba}")
        probability = predicted_proba[predicted_class] * 100
        
        if predicted_class == 1:
            st.success(f"✅ 预测结果：麻附益肾方治疗对您可能有效 (概率: {probability:.1f}%)")
            st.info("""
            **建议：**
            - 按照医嘱继续使用麻附益肾方治疗
            - 定期复查尿蛋白和肾功能指标
            - 保持低盐低脂饮食
            """)
        else:
            st.warning(f"⚠️ 预测结果：麻附益肾方治疗对您可能效果有限 (概率: {probability:.1f}%)")
            st.info("""
            **建议：**
            - 咨询医生是否需要调整治疗方案
            - 考虑结合其他治疗方式
            - 密切监测病情变化
            """)
        
        # 添加概率条可视化
        prob_df = pd.DataFrame({
            '治疗有效': [predicted_proba[1]],
            '治疗无效': [predicted_proba[0]]
        })
        st.bar_chart(prob_df)
        
        # SHAP解释
        st.subheader("特征影响分析")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_df)
        
        # 使用Matplotlib创建SHAP图
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, input_df, plot_type="bar", show=False)
        plt.tight_layout()
        st.pyplot(plt)
        
        # 特征重要性说明
        st.caption("""
        **说明：**
        - 上图显示了各个特征对预测结果的影响程度
        - 正值表示增加治疗有效的概率
        - 负值表示增加治疗无效的概率
        """)
        
    except Exception as e:
        st.error(f"发生错误: {str(e)}")
        st.info("请检查模型文件是否匹配当前特征设置")
