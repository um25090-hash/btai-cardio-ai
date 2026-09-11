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
# -------------------------------------------------
# BTAI CARDIOLOGY AI ASSISTANT
# -------------------------------------------------

from google import genai
from google.genai import types

st.markdown("---")
st.header("💬 BTAI Cardiology AI Assistant")

st.write(
    "Ask me about cardiology services, common heart-related tests, "
    "how to prepare for an appointment, and when symptoms may require urgent care."
)

st.info(
    "This assistant provides general educational and hospital-navigation information. "
    "It does not diagnose conditions, prescribe medicines, or replace a doctor."
)

# Create Gemini client using the secret stored in Streamlit
client = genai.Client(
    api_key=st.secrets["GEMINI_API_KEY"]
)

# Create chat history the first time the app loads
if "cardio_messages" not in st.session_state:
    st.session_state.cardio_messages = [
        {
            "role": "assistant",
            "content": (
                "Hello! I am the BTAI Cardiology AI Assistant. "
                "I can help you understand cardiology services, common tests, "
                "appointment preparation, and general heart-health information. "
                "How can I help you today?"
            )
        }
    ]

# Display previous chat messages
for message in st.session_state.cardio_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input box
user_question = st.chat_input(
    "Ask a cardiology-related question..."
)

if user_question:

    # Show and save user's question
    st.session_state.cardio_messages.append(
        {
            "role": "user",
            "content": user_question
        }
    )

    with st.chat_message("user"):
        st.markdown(user_question)

    # System instructions for safe patient-facing behavior
    system_prompt = """
You are BTAI Cardiology AI Assistant, a patient-facing hospital care navigator.

Your role is to:
- explain common cardiology services and tests in simple language
- help patients understand which type of cardiology service may be appropriate
- help patients prepare for a cardiology appointment
- explain common heart-health terminology
- provide general preventive heart-health information
- encourage professional medical care when appropriate

You must NOT:
- diagnose a disease
- claim that a patient definitely has or does not have a condition
- prescribe medicines
- recommend changing or stopping medication
- provide medication doses
- replace a cardiologist or emergency service

EMERGENCY RULE:
If the user mentions severe or persistent chest pain, severe difficulty breathing,
fainting, collapse, new severe weakness, or other potentially life-threatening
symptoms, clearly tell them to seek emergency medical care immediately.

Use calm, simple, patient-friendly language.
Keep answers concise unless the patient asks for more detail.
When useful, recommend the type of service they may want to discuss with the hospital,
such as cardiology consultation, ECG, echocardiography, preventive cardiac check-up,
or emergency care.

Always remind the patient that the information is general and not a diagnosis when
the question involves symptoms or an individual medical situation.
"""

    # Include recent conversation so Gemini remembers the discussion
    conversation_text = ""

    for message in st.session_state.cardio_messages[-8:]:
        conversation_text += (
            f"{message['role'].upper()}: {message['content']}\n"
        )

    try:
        response = client.models.generate_content(
            model="gemini-3.1-flash-lite",
            contents=conversation_text,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.3
            )
        )

        ai_answer = response.text

    except Exception as e:
        ai_answer = (
            "I am temporarily unable to answer. "
            "Please try again in a moment."
        )

    # Save and display AI answer
    st.session_state.cardio_messages.append(
        {
            "role": "assistant",
            "content": ai_answer
        }
    )

    with st.chat_message("assistant"):
        st.markdown(ai_answer)
