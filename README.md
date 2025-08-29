# Heart Disease Prediction using Machine Learning and Explainable AI Techniques

This is a heart disease prediction system using Machine Learning and Explianble AI Techniques. Xgboost has been used for the prediction, along with SHAP (Shapley Additive eXplanations) feature, SHAP extracts the most important features which helped in the prediction. A patient can reduce the risk of heart disease if he/she focus on those features and take care of their health accordingly.

For explaining the prediction made by Xgboost along with the SHAP values, the phi-3 model from ollama has been used. The phi model takes into account the prediction and shap values and explains why that particular prediciton being made (yes/no). And it also gives recommendations on how to improve their health and which factors to focus more on.

---

## 📂 Project Structure

streamlit-ml-app/
│
├── app.py                      # Entry point for Streamlit UI
│
├── backend/
│   ├── __init__.py
│   ├── prediction_system.py    # ML model loading + predictions
│   ├── llm_service.py          # Handles LLM (Ollama) explanation
│   └── utils.py                # Shared helpers (hashing, etc.)
│
├── models/
│   ├── scaler.joblib
│   ├── shap_tree_explainer.joblib
│   └── xgb_model.joblib
│
├── requirements.txt
├── .gitignore
└── README.md

---

## ⚡ Features

- Interactive frontend with Streamlit
- Prediction system powered by XGBoost
- Model interpretability with SHAP
- LLM explanations via Ollama (phi-3)
- Reusable utilities for hashing, preprocessing.

---

## 🛠️ Installation & Setup

### 1 Clone the repository
git clone https://github.com//heart_project.git
cd heart_project

### 2 Create a virtual environment
python -m venv .venv

### 3 Activate the environment
- Windows
  .venv\Scripts\activate
- Linux/Mac
  source .venv/bin/activate

### 4 Install dependencies
pip install -r requirements.txt

---

## Running the App

Run the Streamlit app:
streamlit run app.py

This will start a local server. Open the provided URL in your browser.
While registering please enter the
email:- heartprediction@gmail.com
password:- admin@123

---

## Models

The models/ folder already contains:
- scaler.joblib → Preprocessing scaler
- xgb_model.joblib → Trained XGBoost model
- shap_tree_explainer.joblib → SHAP explainer

---

## Contributing

Contributions are welcome! Please fork the repo and submit a pull request.

---
=======
