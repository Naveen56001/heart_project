import joblib
import pandas as pd
import shap
from .llm_service import LLMService

class PredictionSystem:
    def __init__(self):
        # Load trained models
        self.model = joblib.load("models/xgb_model.joblib")
        self.scaler = joblib.load("models/scaler.joblib")
        self.explainer = joblib.load("models/shap_tree_explainer.joblib")
        
        # Feature names (must match training data)
        self.feature_names = [
            "age", "sex", "chest pain type", "resting bp s", 
            "cholesterol", "fasting blood sugar", "resting ecg",
            "max heart rate", "exercise angina", "oldpeak", "ST slope"
        ]
        self.llm = LLMService()

    def predict(self, input_data: dict) -> dict:
        """Process input and return prediction with explanation"""
        input_df = pd.DataFrame([input_data])
        scaled_input = self.scaler.transform(input_df)

        # Prediction
        pred = self.model.predict(scaled_input)[0]
        proba = self.model.predict_proba(scaled_input)[0][1]

        # SHAP
        shap_values = self.explainer(scaled_input)
        top_features = sorted(
            zip(self.feature_names, shap_values.values[0]),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:3]

        # LLM Explanation
        explanation = self.llm.explain(pred, proba, top_features)

        return {
            "prediction": pred,
            "probability": proba,
            "top_features": top_features,
            "explanation": explanation
        }
