import streamlit as st

st.title("ทดสอบ Dashboard")
T = st.slider("อุณหภูมิ Lye ขาเข้า (°C)", 35, 50, 42)
st.write("คุณเลือก", T, "°C")

