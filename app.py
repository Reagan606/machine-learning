

import streamlit as st
import sys, platform
st.title("Health Check")
st.write("Python:", sys.version)
import streamlit as st; st.write("Streamlit:", st.__version__)

# 依赖自检（按需增删）
import numpy, pandas, sklearn
st.write("numpy:", numpy.__version__)
st.write("pandas:", pandas.__version__)
st.write("sklearn:", sklearn.__version__)

# 简单绘图（无 plt.show()）
import matplotlib.pyplot as plt
fig = plt.figure()
plt.plot([0,1,2],[0,1,0])
st.pyplot(fig)

st.success("Environment OK ✅")
