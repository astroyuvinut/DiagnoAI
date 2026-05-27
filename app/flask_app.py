import os
import json
from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import joblib

from src.explainability.shap_lime import load_artifacts, build_lime_explainer, lime_explanation_for_input

app = Flask(__name__)

# Global variables
model = None
preprocessor = None
feature_names = []
lime_explainer = None

def load_model():
    global model, preprocessor, feature_names, lime_explainer
    try:
        model, preprocessor, feature_names, background_X = load_artifacts()
        lime_explainer = build_lime_explainer(background_X, feature_names)
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

@app.route('/')
def index():
    return render_template('index.html', features=feature_names)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Get input data
        data = request.json
        input_values = [float(data.get(f, 0)) for f in feature_names]
        
        # Create DataFrame
        X_df = pd.DataFrame([input_values], columns=feature_names)
        
        # Preprocess
        x_pre = preprocessor.transform(X_df)
        if hasattr(x_pre, "toarray"):
            x_pre = x_pre.toarray()
        
        # Predict
        proba = float(model.predict_proba(x_pre)[:, 1][0])
        prediction = "Positive (disease)" if proba >= 0.5 else "Negative"
        
        # LIME explanation
        lime_list, lime_score = lime_explanation_for_input(
            lime_explainer, model.named_steps["clf"], np.asarray(x_pre)
        )
        
        return jsonify({
            'prediction': prediction,
            'probability': proba,
            'lime_explanation': lime_list,
            'lime_score': lime_score
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

if __name__ == '__main__':
    if load_model():
        print("Model loaded successfully!")
        app.run(host='0.0.0.0', port=5000, debug=True)
    else:
        print("Failed to load model. Please run training first.")
