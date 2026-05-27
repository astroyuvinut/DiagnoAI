# DiagnoAI - Healthcare Predictive Analytics (Breast Cancer MVP)

## Quickstart (Windows PowerShell)

1) Create venv and install:
```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

2) Train models and save artifacts:
```bash
python -m src.models.train
```
- Saves artifacts to `artifacts/`:
  - `model.joblib`, `preprocessor.joblib`, `feature_names.joblib`
  - `background_X.npy` (for SHAP/LIME), `metrics.json`, `roc_*.png`

3) Launch Gradio app:
```bash
python -m app.gradio_app
```
- Open the local URL shown in the console.
- Enter feature values; get risk probability with SHAP and LIME explanations.

## Stack
- Dataset: `sklearn.datasets.load_breast_cancer`
- Models: Logistic Regression, Random Forest, XGBoost (GridSearch on recall)
- Explainability: SHAP (global/local), LIME (local)
- Metrics: Accuracy, Precision, Recall, F1, ROC-AUC + confusion matrix (console)

## Next Ideas
- Add other diseases/datasets (diabetes, heart disease)
- Handle imbalance (class weights/SMOTE)
- Add fairness checks and monitoring
- Deploy to cloud, add API endpoints
