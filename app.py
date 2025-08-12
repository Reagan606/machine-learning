import streamlit as st, sys, platform
st.title("Health Check")
st.write("Python:", sys.version)
import streamlit as s; st.write("Streamlit:", s.__version__)
import numpy, pandas, sklearn
st.write("numpy:", numpy.__version__)
st.write("pandas:", pandas.__version__)
st.write("sklearn:", sklearn.__version__)
import matplotlib.pyplot as plt
fig = plt.figure()
plt.plot([0,1,2],[0,1,0])
st.pyplot(fig)
st.success("Environment OK ✅")
