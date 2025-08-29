import streamlit as st
import pandas as pd
import time
import shap
import matplotlib.pyplot as plt

from backend.prediction_system import PredictionSystem
from backend.utils import hash_password

# ----------------- Session State Setup -----------------
if 'users' not in st.session_state:
    st.session_state.users = {
        "admin": {"email": "heartprediction@gmail.com", "password": hash_password("admin@123")},
        "doctor": {"email": "doctor@example.com", "password": hash_password("Suresh@123")}
    }

if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

if 'page' not in st.session_state:
    st.session_state.page = "register"

# ----------------- Load Model System -----------------
@st.cache_resource
def load_system():
    return PredictionSystem()

system = load_system()

# ----------------- Pages -----------------
def registration_page():
    st.title("Register")
    username = st.text_input("Username")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    confirm = st.text_input("Confirm Password", type="password")

    if st.button("Register"):
        if password != confirm:
            st.error("Passwords do not match")
        elif username in st.session_state.users:
            st.error("Username already exists")
        else:
            st.session_state.users[username] = {"email": email, "password": hash_password(password)}
            st.success("Registration successful! Go to login.")

    if st.button("Go to Login"):
        st.session_state.page = "login"

def login_page():
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        user = st.session_state.users.get(username)
        if user and user["password"] == hash_password(password):
            st.session_state.authenticated = True
            st.session_state.username = username
            st.session_state.page = "predict"
        else:
            st.error("Invalid username or password")

    if st.button("Go to Registration"):
        st.session_state.page = "register"

def prediction_page():
    st.title("Heart Disease Risk Predictor")
    st.markdown(f"**Logged in as: `{st.session_state.username}`**")

    if st.button("Logout"):
        st.session_state.authenticated = False
        st.session_state.page = "login"
        return

    with st.form("patient_input"):
        st.header("Patient Information")
        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input("Age", min_value=20, max_value=100, value=50)
            sex = st.selectbox("Sex", ["Male", "Female"])
            chest_pain_type = st.selectbox("Chest Pain Type", [
                "Typical angina", "Atypical angina", "Non-anginal pain", "Asymptomatic"
            ])
            resting_bp_s = st.number_input("Resting BP (mm Hg)", 90, 200, 120)
            cholesterol = st.number_input("Cholesterol (mg/dl)", 100, 600, 200)

        with col2:
            fasting_blood_sugar = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
            resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"])
            max_heart_rate = st.number_input("Max Heart Rate", 70, 220, 150)
            exercise_angina = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
            oldpeak = st.number_input("ST Depression (oldpeak)", 0.0, 6.2, 1.0)
            ST_slope = st.selectbox("ST Segment Slope", ["Upsloping", "Flat", "Downsloping"])

        submitted = st.form_submit_button("Predict Risk")

    if submitted:
        input_data = {
            "age": age,
            "sex": 1 if sex == "Male" else 0,
            "chest pain type": ["Typical angina", "Atypical angina", "Non-anginal pain", "Asymptomatic"].index(chest_pain_type),
            "resting bp s": resting_bp_s,
            "cholesterol": cholesterol,
            "fasting blood sugar": 1 if fasting_blood_sugar == "Yes" else 0,
            "resting ecg": ["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"].index(resting_ecg),
            "max heart rate": max_heart_rate,
            "exercise angina": 1 if exercise_angina == "Yes" else 0,
            "oldpeak": oldpeak,
            "ST slope": ["Upsloping", "Flat", "Downsloping"].index(ST_slope)
        }

        with st.spinner("Analyzing patient data... It may take a minute..."):
            start_time = time.time()
            result = system.predict(input_data)
            elapsed = time.time() - start_time

        st.success(f"Analysis completed in {elapsed:.2f} seconds")
        st.header("Results")
        risk_color = "red" if result["prediction"] == 1 else "green"
        st.markdown(f"""
        ### <span style='color:{risk_color}'>Prediction: {'High Risk' if result['prediction'] == 1 else 'Low Risk'}</span>
        Probability: {result['probability']:.1%}
        """, unsafe_allow_html=True)

        st.subheader("Key Contributing Factors")
        for feat, val in result["top_features"]:
            direction = "↑ Increased risk" if val > 0 else "↓ Decreased risk"
            st.write(f"- **{feat}**: {abs(val):.3f} ({direction})")
        
        # -------------- SHAP Force Plot ---------------- #
        scaled_input = system.scaler.transform(pd.DataFrame([input_data]))
        shap_values = system.explainer(scaled_input)

        plt.figure(figsize=(12, 5))
        shap.plots.force(
            base_value=system.explainer.expected_value,
            shap_values=shap_values.values[0],
            features=scaled_input[0],
            feature_names=system.feature_names,
            matplotlib=True,
            show=False,
            contribution_threshold=0.05,
            text_rotation=0
        )

        ax = plt.gca()
        ax.tick_params(axis='both', which='major', labelsize=9)
        for text in ax.texts:
            if '=' in text.get_text():
                parts = text.get_text().split('=')
                if len(parts) == 2:
                    try:
                        value = float(parts[1])
                        parts[1] = f"{value:.2f}"
                        text.set_text('='.join(parts))
                    except ValueError:
                        pass

        plt.tight_layout(pad=2.0)
        st.subheader("SHAP Force Plot")
        st.pyplot(plt)
        plt.close()

        st.subheader("Medical Explanation")
        st.write(result["explanation"])

# ----------------- Router -----------------
if not st.session_state.authenticated:
    if st.session_state.page == "register":
        registration_page()
    elif st.session_state.page == "login":
        login_page()
else:
    prediction_page()
