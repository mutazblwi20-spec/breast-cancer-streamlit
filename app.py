import streamlit as st
import numpy as np
import joblib

# إعداد الصفحة
st.set_page_config(
    page_title="Breast Cancer Predictor",
    layout="centered"
)

# تحميل الموديل و الـ scaler
model = joblib.load("breast_cancer_model.pkl")
scaler = joblib.load("scaler.pkl")

# عنوان التطبيق
st.title("🩺 Breast Cancer Prediction App")
st.markdown("أدخل القيم التالية لتوقع نوع الورم")

# أسماء الخصائص (30 feature – نفس الموديل)
feature_names = [
    "radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean",
    "compactness_mean","concavity_mean","concave points_mean","symmetry_mean","fractal_dimension_mean",
    "radius_se","texture_se","perimeter_se","area_se","smoothness_se",
    "compactness_se","concavity_se","concave points_se","symmetry_se","fractal_dimension_se",
    "radius_worst","texture_worst","perimeter_worst","area_worst","smoothness_worst",
    "compactness_worst","concavity_worst","concave points_worst","symmetry_worst","fractal_dimension_worst"
]

# إدخال القيم من المستخدم
inputs = []

for feature in feature_names:
    value = st.number_input(feature, value=0.0)
    inputs.append(value)

# زر التوقع
if st.button("Predict"):
    input_array = np.array(inputs).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.error("⚠️ Malignant (ورم خبيث)")
    else:
        st.success("✅ Benign (ورم حميد)")
