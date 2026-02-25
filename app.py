import streamlit as st
import pickle
import numpy as np

# تحميل النموذج و scaler
model, scaler = pickle.load(open("breast_cancer_model.pkl", "rb"))

st.set_page_config(page_title="Breast Cancer Predictor", layout="centered")

st.title("🩺 Breast Cancer Prediction App")
st.markdown("أدخل القيم التالية لتوقع نوع الورم")

# إنشاء حقول إدخال ديناميكية
feature_names = [
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
    "compactness_mean", "concavity_mean", "concave points_mean", "symmetry_mean", "fractal_dimension_mean"
]

inputs = []

for feature in feature_names:
    value = st.number_input(f"{feature}", value=0.0)
    inputs.append(value)

if st.button("Predict"):
    input_array = np.array(inputs).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.error("⚠️ Malignant (ورم خبيث)")
    else:
        st.success("✅ Benign (ورم حميد)")