import ollama

class LLMService:
    def __init__(self, model="phi"):
        self.model = model

    def explain(self, pred, proba, top_features):
        """Generate explanation using LLM"""
        prompt = self._build_prompt(pred, proba, top_features)
        response = ollama.chat(
            model=self.model,
            messages=[{'role': 'user', 'content': prompt}],
            options={'temperature': 0.3}
        )
        return response['message']['content']

    def _build_prompt(self, pred, proba, top_features):
        return f"""Explain the SHAP values and prediction for a heart disease patient.
Prediction: {'High risk' if pred == 1 else 'Low risk'} ({proba:.1%})
Top Factors:
""" + "\n".join([
            f"  {feat}: {val:.3f} (↑risk)" if val > 0 else f"  {feat}: {abs(val):.3f} (↓risk)"
            for feat, val in top_features
        ])