import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="BTAI CardioAI",
    page_icon="❤️",
    layout="centered"
)

# Load trained model
model = joblib.load("btai_cardio_model.pkl")
feature_names = joblib.load("feature_names.pkl")

st.title("❤️ BTAI CardioAI")

st.subheader("AI-Assisted Cardiovascular Assessment")

st.warning(
    "Educational demonstration only. "
    "This tool is not intended for medical diagnosis or treatment decisions."
)

st.markdown("---")

st.header("Patient Information")

age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=50
)

sex_display = st.selectbox(
    "Sex",
    ["Female", "Male"]
)
sex = 0 if sex_display == "Female" else 1

cp_display = st.selectbox(
    "Chest Pain Type",
    [
        "Typical angina",
        "Atypical angina",
        "Non-anginal pain",
        "Asymptomatic"
    ]
)

cp_map = {
    "Typical angina": 1,
    "Atypical angina": 2,
    "Non-anginal pain": 3,
    "Asymptomatic": 4
}
cp = cp_map[cp_display]

trestbps = st.number_input(
    "Resting Blood Pressure (mmHg)",
    min_value=70,
    max_value=250,
    value=120
)

chol = st.number_input(
    "Serum Cholesterol (mg/dL)",
    min_value=80,
    max_value=700,
    value=200
)

fbs_display = st.selectbox(
    "Fasting Blood Sugar > 120 mg/dL?",
    ["No", "Yes"]
)
fbs = 1 if fbs_display == "Yes" else 0

restecg_display = st.selectbox(
    "Resting ECG",
    [
        "Normal",
        "ST-T wave abnormality",
        "Left ventricular hypertrophy"
    ]
)

restecg_map = {
    "Normal": 0,
    "ST-T wave abnormality": 1,
    "Left ventricular hypertrophy": 2
}
restecg = restecg_map[restecg_display]

thalach = st.number_input(
    "Maximum Heart Rate Achieved",
    min_value=50,
    max_value=230,
    value=150
)

exang_display = st.selectbox(
    "Exercise-Induced Angina?",
    ["No", "Yes"]
)
exang = 1 if exang_display == "Yes" else 0

oldpeak = st.number_input(
    "ST Depression (Oldpeak)",
    min_value=0.0,
    max_value=10.0,
    value=1.0,
    step=0.1
)

slope = st.selectbox(
    "ST Segment Slope",
    [1, 2, 3],
    format_func=lambda x: {
        1: "Upsloping",
        2: "Flat",
        3: "Downsloping"
    }[x]
)

ca = st.number_input(
    "Number of Major Vessels",
    min_value=0,
    max_value=3,
    value=0
)

thal_display = st.selectbox(
    "Thal Test",
    [
        "Normal",
        "Fixed defect",
        "Reversible defect"
    ]
)

thal_map = {
    "Normal": 3,
    "Fixed defect": 6,
    "Reversible defect": 7
}
thal = thal_map[thal_display]

if st.button(
    "❤️ Analyse Cardiac Pattern",
    use_container_width=True
):

    input_data = {
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal
    }

    patient = pd.DataFrame(
        [[input_data[name] for name in feature_names]],
        columns=feature_names
    )

    prediction = model.predict(patient)[0]
    probability = model.predict_proba(patient)[0][1]

    st.markdown("---")
    st.header("AI Assessment")

    st.metric(
        "Model Probability",
        f"{probability * 100:.1f}%"
    )

    if prediction == 1:
        st.error("Elevated heart-disease pattern detected.")
    else:
        st.success("Lower heart-disease pattern detected.")

    st.progress(float(probability))

    st.caption(
        "This percentage is the model's statistical classification probability "
        "based on its training data, not a clinically validated personal risk score."
    )

st.markdown("---")

st.caption(
    "BTAI Hospital CardioAI • Educational AI Demonstration"
)
