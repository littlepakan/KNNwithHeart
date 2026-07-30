from sklearn.neighbors import KNeighborsClassifier
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

st.title('การทำนายข้อมูลโรคหัวใจด้วยเทคนิค K-Nearest Neighbor')
#st.image("./img/kairung.jpg")
col1, col2 = st.columns(2)

with col1:
   st.header("")
   st.image("./img/heart1.jpg")

with col2:
   st.header("")
   st.image("./img/heart2.jpg")


html_7 = """
<div style="background-color:#33beff;padding:15px;border-radius:15px 15px 15px 15px;border-style:'solid';border-color:black">
<center><h4>ข้อมูลโรคหัวใจสำหรับทำนาย</h4></center>
</div>
"""
st.markdown(html_7, unsafe_allow_html=True)
st.markdown("")
st.markdown("")

st.subheader("ข้อมูลส่วนแรก 10 แถว")
dt = pd.read_csv("./data/Heart3.csv")
st.write(dt.head(10))
st.subheader("ข้อมูลส่วนสุดท้าย 10 แถว")
st.write(dt.tail(10))

# สถิติพื้นฐาน
st.subheader("📈 สถิติพื้นฐานของข้อมูล")
st.write(dt.describe())

# การเลือกแสดงกราฟตามฟีเจอร์
st.subheader("📌 เลือกฟีเจอร์เพื่อดูการกระจายข้อมูล")
feature = st.selectbox("เลือกฟีเจอร์", dt.columns[:-1])

# วาดกราฟ boxplot
st.write(f"### 🎯 Boxplot: {feature} แยกตามชนิดของโรคหัวใจ")
fig, ax = plt.subplots()
sns.boxplot(data=dt, x='HeartDisease', y=feature, ax=ax)
st.pyplot(fig)

# วาด pairplot
if st.checkbox("แสดง Pairplot (ใช้เวลาประมวลผลเล็กน้อย)"):
    st.write("### 🌺 Pairplot: การกระจายของข้อมูลทั้งหมด")
    fig2 = sns.pairplot(dt, hue='HeartDisease')
    st.pyplot(fig2)

html_8 = """
<div style="background-color:#6BD5DA;padding:15px;border-radius:15px 15px 15px 15px;border-style:'solid';border-color:black">
<center><h5>ทำนายข้อมูล</h5></center>
</div>
"""
st.markdown(html_8, unsafe_allow_html=True)
st.markdown("")

cols = dt.drop('HeartDisease', axis=1).columns

# ============================================
# ส่วนรับข้อมูลสำหรับทำนายผล (แทนที่ก้อน A1 - A11 เดิม)
# ============================================

col1, col2, col3 = st.columns(3)
with col1:
    # A1: Age
    A1 = st.number_input("🎂 อายุ (ปี)", min_value=20, max_value=100, value=55)

with col2:
    # A2: Sex
    sex_opt = st.selectbox(
        "⚧ เพศ",
        options=[(1, "ชาย"), (0, "หญิง")],
        format_func=lambda x: x[1]
    )
    A2 = sex_opt[0]

with col3:
    # A3: ChestPainType
    cp_opt = st.selectbox(
        "💔 อาการเจ็บหน้าอก",
        options=[(1, "ATA"), (2, "NAP"), (3, "ASY"), (4, "TA")],
        format_func=lambda x: x[1]
    )
    A3 = cp_opt[0]

col4, col5, col6 = st.columns(3)
with col4:
    # A4: RestingBP
    A4 = st.number_input("🩸 ความดันโลหิต (mmHg)", min_value=80, max_value=200, value=130)

with col5:
    # A5: Cholesterol
    A5 = st.number_input("🧪 โคเลสเตอรอล (mg/dl)", min_value=0, max_value=600, value=220)

with col6:
    # A6: FastingBS
    fbs_opt = st.selectbox(
        "🍬 น้ำตาลตอนอดอาหาร > 120 mg/dl",
        options=[(0, "ไม่"), (1, "ใช่")],
        format_func=lambda x: x[1]
    )
    A6 = fbs_opt[0]

col7, col8, col9 = st.columns(3)
with col7:
    # A7: RestingECG
    ecg_opt = st.selectbox(
        "📊 ผล ECG (RestingECG)",
        options=[(0, "Normal"), (1, "Abnormal"), (2, "Hypertrophy")],
        format_func=lambda x: x[1]
    )
    A7 = ecg_opt[0]

with col8:
    # A8: MaxHR
    A8 = st.number_input("💓 อัตราการเต้นหัวใจสูงสุด (Max HR)", min_value=60, max_value=220, value=140)

with col9:
    # A9: ExerciseAngina
    exang_opt = st.selectbox(
        "🏃 เจ็บหน้าอกขณะออกกำลังกาย",
        options=[(0, "ไม่"), (1, "ใช่")],
        format_func=lambda x: x[1]
    )
    A9 = exang_opt[0]

col10, col11 = st.columns(2)
with col10:
    # A10: Oldpeak
    A10 = st.number_input("📉 Oldpeak (ST Depression)", min_value=-3.0, max_value=7.0, value=1.0, step=0.1)

with col11:
    # A11: ST_Slope
    slope_opt = st.selectbox(
        "📈 ST Slope",
        options=[(1, "Upsloping"), (2, "Flat"), (3, "Downsloping")],
        format_func=lambda x: x[1]
    )
    A11 = slope_opt[0]

if st.button("ทำนายผล"):
   #st.write("ทำนาย")
   #dt = pd.read_csv("./data/iris-3.csv") 
   X = dt.drop('HeartDisease', axis=1)
   y = dt.HeartDisease

   Knn_model = KNeighborsClassifier(n_neighbors=3)
   Knn_model.fit(X, y)  
    
   x_input = np.array([[A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A11]])
   st.write(Knn_model.predict(x_input))
   
   out=Knn_model.predict(x_input)

   if out[0] == 1:
    st.image("./img/heart1.jpg")
   else:
    st.image("./img/heart2.jpg")
else:
    st.write("ไม่ทำนาย")
    st.write("test")