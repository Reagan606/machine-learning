import streamlit as st, traceback

def main():
    st.set_page_config(page_title="App", layout="wide")
    st.write("App starting...")
    # ===== 你的原始代码从这里粘进去 =====

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error("未处理异常：")
        st.exception(e)
        st.code(traceback.format_exc())



import streamlit as st
import numpy as np
import pandas as pd
import traceback

# 可选：如果你确定在用 xgboost + shap
import shap
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
import joblib
import os

st.set_page_config(page_title="麻附益肾方治疗膜性肾病疗效预测", layout="centered")

# ========== 安全加载模型：优先 JSON（版本安全），否则回退 joblib ==========
@st.cache_resource
def load_model():
    # 推荐：用 XGBClassifier().load_model("model_xgb.json")
    if os.path.exists("model_xgb.json"):
        clf = XGBClassifier()
        clf.load_model("model_xgb.json")
        return clf
    # 回退：加载 pkl（可能有版本警告）
    elif os.path.exists("xgboost_model_15_features.pkl"):
        return joblib.load("xgboost_model_15_features.pkl")
    else:
        raise FileNotFoundError("缺少模型文件：请提供 model_xgb.json 或 xgboost_model_15_features.pkl")

try:
    model = load_model()
except Exception as e:
    st.error("加载模型失败：")
    st.exception(e)
    st.stop()

# ========== 选项与特征 ==========
Diagnose_options = {
    1: "特发",
    2: "不典型",
    3: "继发",
    4: "抗体诊断",
    5: "未知",
}

# 注意：列名要与训练时完全一致（含顺序）
feature_names = [
    "TBA","PA","RBC_U","CO2","CG","CVRDW",
    "Lymph_abs","Baso_abs","ALP","TC","Eos_abs","MPV",
    "Diagnose","β2_MG","LDL_C"
]

st.title("麻附益肾方治疗膜性肾病疗效预测")

# ========== 输入区 ==========
TBA = st.number_input("总胆汁酸 TBA（μmol/L）", min_value=0.0, max_value=200.0, value=3.0, format="%.1f")
PA = st.number_input("前白蛋白 PA（mg/L）", min_value=0.0, max_value=1000.0, value=300.0, format="%.1f")
RBC_U = st.number_input("红细胞（尿液）（个/HPF）", min_value=0.00, max_value=1_000.00, value=10.00, format="%.2f")
CO2 = st.number_input("二氧化碳 CO₂（mmol/L）", min_value=0.0, max_value=60.0, value=25.0, format="%.1f")
CG = st.number_input("甘胆酸 CG（mg/L）", min_value=0.00, max_value=100.00, value=1.00, format="%.2f")
CVRDW = st.number_input("红细胞体积分布宽度 CVRDW（%）", min_value=0.0, max_value=50.0, value=12.0, format="%.1f")
Lymph_abs = st.number_input("淋巴细胞绝对值 Lymph#（×10⁹/L）", min_value=0.00, max_value=20.00, value=2.50, format="%.2f")
Baso_abs = st.number_input("嗜碱性粒细胞绝对值 Baso#（×10⁹/L）", min_value=0.00, max_value=1.00, value=0.10, format="%.2f")
ALP = st.number_input("碱性磷酸酶 ALP（IU/L）", min_value=0.0, max_value=2000.0, value=110.0, format="%.1f")
TC = st.number_input("总胆固醇 TC（mmol/L）", min_value=0.00, max_value=30.00, value=4.50, format="%.2f")
Eos_abs = st.number_input("嗜酸性粒细胞绝对值 Eos#（×10⁹/L）", min_value=0.00, max_value=5.00, value=0.10, format="%.2f")
MPV = st.number_input("平均血小板体积 MPV（fL）", min_value=0.0, max_value=20.0, value=10.5, format="%.1f")
Diagnose = st.selectbox("具体诊断", options=list(Diagnose_options.keys()),
                        format_func=lambda x: Diagnose_options[x])
beta2_MG = st.number_input("β2 微球蛋白 β2-MG（mg/L）", min_value=0.00, max_value=50.00, value=1.00, format="%.2f")
LDL_C = st.number_input("低密度脂蛋白胆固醇 LDL-C（mmol/L）", min_value=0.00, max_value=20.00, value=3.00, format="%.2f")

# 组装为 DataFrame（确保与训练一致的列名与顺序）
row = [TBA, PA, RBC_U, CO2, CG, CVRDW, Lymph_abs, Baso_abs, ALP, TC, Eos_abs, MPV, Diagnose, beta2_MG, LDL_C]
X = pd.DataFrame([row], columns=feature_names)

# 如训练时 Diagnose 是类别编码（整数），保留为 int；若当时做了独热编码，需对应改造
X["Diagnose"] = X["Diagnose"].astype(int)

# ========== 预测 + SHAP ==========
def predict_and_explain(model, X: pd.DataFrame):
    pred_proba = None
    pred_class = None
    # 预测
    if hasattr(model, "predict_proba"):
        pred_proba = model.predict_proba(X)[0]
        pred_class = int(np.argmax(pred_proba))
    else:
        y_hat = model.predict(X)
        pred_class = int(y_hat[0])

    # 显示结果
    st.subheader("预测结果")
    st.write(f"**Predicted Class:** {pred_class}")
    if pred_proba is not None:
        st.write("**Prediction Probabilities:**", np.round(pred_proba, 4))
        probability = float(pred_proba[pred_class] * 100.0)
    else:
        probability = None

    if pred_class == 1:
        st.success(f"预测：麻附益肾方治疗**可能有效**" + (f"（概率 {probability:.1f}%）" if probability is not None else ""))
        st.markdown(
            "- 按医嘱继续治疗\n"
            "- 定期复查尿蛋白和肾功能\n"
            "- 保持低盐低脂饮食"
        )
    else:
        st.warning(f"预测：麻附益肾方治疗**可能效果有限**" + (f"（概率 {probability:.1f}%）" if probability is not None else ""))
        st.markdown(
            "- 咨询医生是否调整方案\n"
            "- 可考虑联合其他治疗\n"
            "- 密切监测病情变化"
        )

    # SHAP 解释（树模型）
    try:
        st.subheader("特征贡献（SHAP）")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        # 兼容 shap 返回格式：二分类可能返回 np.ndarray 或 list
        if isinstance(shap_values, list):
            # 取正类的 shap 值（通常为第1类），如与你训练时定义相反，可切换索引
            sv = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        else:
            sv = shap_values

        # force_plot（matplotlib）直接在页面输出，而不是先存 PNG 再读
        plt.figure(figsize=(10, 1.8))
        shap.force_plot(explainer.expected_value if np.isscalar(explainer.expected_value) else explainer.expected_value[1],
                        sv[0], X.iloc[0], matplotlib=True, show=False)
        st.pyplot(plt.gcf(), clear_figure=True)

        # 也给一个 bar 图（更稳）
        plt.figure(figsize=(8, 5))
        shap_values_for_bar = sv[0] if sv.ndim == 2 else sv
        shap.summary_plot(shap_values_for_bar.reshape(1, -1), X, plot_type="bar", show=False)
        st.pyplot(plt.gcf(), clear_figure=True)

    except Exception as e:
        st.info("SHAP 可视化遇到问题（可能与 shap/xgboost 版本有关），显示原始异常以便排查：")
        st.exception(e)
        st.code(traceback.format_exc())

if st.button("Predict"):
    try:
        predict_and_explain(model, X)
    except Exception as e:
        st.error("运行时出现错误：")
        st.exception(e)
        st.code(traceback.format_exc())

