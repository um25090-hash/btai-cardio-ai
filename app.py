import streamlit as st
import pandas as pd
import joblib
import requests

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------

st.set_page_config(
    page_title="BTAI CardioAI",
    page_icon="❤️",
    layout="centered"
)

# -------------------------------------------------
# LOAD HEART-DISEASE MODEL
# -------------------------------------------------

model = joblib.load("btai_cardio_model.pkl")
feature_names = joblib.load("feature_names.pkl")

# -------------------------------------------------
# HEART-RISK PREDICTOR
# -------------------------------------------------

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

st.markdown("---")
st.header("💬 BTAI Cardiology AI Assistant")

st.write(
    "Ask me about cardiology services, common heart-related tests, "
    "appointment preparation, general heart-health information, "
    "and when symptoms may require urgent medical attention."
)

st.info(
    "This assistant provides general educational and hospital-navigation information. "
    "It does not diagnose conditions, prescribe medicines, or replace a doctor."
)

# -------------------------------------------------
# CHAT HISTORY
# -------------------------------------------------

if "cardio_messages" not in st.session_state:
    st.session_state.cardio_messages = [
        {
            "role": "assistant",
            "content": (
                "Hello! I’m the BTAI Cardiology AI Assistant. "
                "I can help explain cardiology tests and services, "
                "help you prepare for an appointment, and answer general "
                "heart-health questions. How can I help you today?"
            )
        }
    ]

for message in st.session_state.cardio_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# -------------------------------------------------
# EMERGENCY SAFETY CHECK
# -------------------------------------------------

def is_emergency(text):
    t = text.lower().replace("’", "'")

    emergency_terms = [
        "severe chest pain",
        "very bad chest pain",
        "terrible chest pain",
        "extreme chest pain",
        "chest painnn",
        "chest is hurting badly",
        "chest hurting badly",
        "cant breathe",
        "can't breathe",
        "cannot breathe",
        "unable to breathe",
        "difficulty breathing",
        "severe breathlessness",
        "gasping for air",
        "fainted",
        "fainting",
        "collapsed",
        "collapse",
        "unconscious",
        "chest pain and sweating",
        "chest pain and breathless",
        "chest pain with breathlessness"
    ]

    return any(term in t for term in emergency_terms)

# -------------------------------------------------
# OPENROUTER SYSTEM PROMPT
# -------------------------------------------------

system_prompt = """
You are the BTAI Cardiology AI Assistant.

You are a patient-facing hospital information and care-navigation assistant.

Your role is to:
- explain common cardiology tests and procedures in simple language
- explain common cardiology services
- help patients understand which kind of cardiology service may be relevant
- help patients prepare for cardiology appointments
- explain common heart-health terminology
- provide general preventive heart-health information
- answer general conversational questions naturally
- be friendly, calm, clear, and concise

Examples of services you may mention:
- Cardiology consultation
- ECG
- Echocardiography
- Holter monitoring
- Preventive cardiac check-up
- Blood pressure evaluation
- Lipid profile review
- Emergency care

You must NOT:
- diagnose a disease
- claim that someone definitely has or does not have a condition
- prescribe medicines
- provide medication dosages
- tell a patient to start, stop, increase, or decrease medicines
- replace a cardiologist
- claim that AI output is a medical diagnosis

If a user describes symptoms:
- provide general guidance only
- encourage professional medical evaluation when appropriate
- remind them that the response is not a diagnosis

If there may be a medical emergency:
- advise immediate emergency medical care

You should also answer normal questions naturally.
For example, if asked your name, say:
"I’m the BTAI Cardiology AI Assistant."

Keep answers patient-friendly and easy to understand.
"""

# -------------------------------------------------
# USER INPUT
# -------------------------------------------------

user_question = st.chat_input(
    "Ask a cardiology-related question..."
)

if user_question:

    st.session_state.cardio_messages.append(
        {
            "role": "user",
            "content": user_question
        }
    )

    with st.chat_message("user"):
        st.markdown(user_question)

    # -------------------------------------------------
    # HARD EMERGENCY OVERRIDE
    # -------------------------------------------------

    if is_emergency(user_question):

        ai_answer = """
🚨 **Please seek emergency medical care immediately.**

Severe chest pain, severe breathing difficulty, fainting, collapse, or similar symptoms can represent a medical emergency.

Please contact your local emergency medical service or go to the nearest emergency department immediately.

Do not rely on this chatbot or the CardioAI predictor to assess an emergency.
"""

    else:

        # -------------------------------------------------
        # BUILD CONVERSATION
        # -------------------------------------------------

        messages = [
            {
                "role": "system",
                "content": system_prompt
            }
        ]

        for message in st.session_state.cardio_messages[-10:]:
            messages.append(
                {
                    "role": message["role"],
                    "content": message["content"]
                }
            )

        # -------------------------------------------------
        # CALL OPENROUTER
        # -------------------------------------------------

        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {st.secrets['OPENROUTER_API_KEY']}",
                    "Content-Type": "application/json",
                    "X-Title": "BTAI Cardiology AI Assistant"
                },
                json={
                    "model": "openrouter/free",
                    "messages": messages,
                    "temperature": 0.3,
                    "max_tokens": 450
                },
                timeout=45
            )

            if response.status_code != 200:
                raise Exception(
                    f"OpenRouter error {response.status_code}: {response.text}"
                )

            data = response.json()

            ai_answer = data["choices"][0]["message"]["content"]

        except Exception as e:
            ai_answer = (
                "I’m temporarily unable to reach the AI service. "
                "Please try again shortly."
            )

    st.session_state.cardio_messages.append(
        {
            "role": "assistant",
            "content": ai_answer
        }
    )

    with st.chat_message("assistant"):
        st.markdown(ai_answer)
