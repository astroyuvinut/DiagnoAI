import os
from typing import List
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gradio as gr

from src.explainability.shap_lime import (
    load_artifacts,
    build_shap_explainer,
    shap_values_for_input,
    build_lime_explainer,
    lime_explanation_for_input,
)

ARTIFACTS_DIR = os.path.join("artifacts")

model = None
preprocessor = None
feature_names: List[str] = []
background_X = None
shap_explainer = None
lime_explainer = None


def load_all():
    global model, preprocessor, feature_names, background_X, shap_explainer, lime_explainer
    model, preprocessor, feature_names, background_X = load_artifacts()
    shap_explainer = build_shap_explainer(model, background_X)
    lime_explainer = build_lime_explainer(background_X, feature_names)


if os.path.exists(os.path.join(ARTIFACTS_DIR, "model.joblib")):
    load_all()


def _shap_chart(shap_vals: np.ndarray, names: List[str]) -> plt.Figure:
    if shap_vals.ndim > 1:
        shap_vals = shap_vals.flatten()
    idx = np.argsort(np.abs(shap_vals))[-12:]
    vals = shap_vals[idx]
    labs = [names[i] for i in idx]
    colors = ["#e74c3c" if v > 0 else "#3498db" for v in vals]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(labs, vals, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP value (impact on prediction)")
    ax.set_title("Top feature contributions (SHAP)")
    fig.tight_layout()
    return fig


def _lime_chart(lime_list: list) -> plt.Figure:
    lime_list = sorted(lime_list, key=lambda x: abs(x[1]))[-12:]
    labs = [x[0] for x in lime_list]
    vals = [x[1] for x in lime_list]
    colors = ["#e74c3c" if v > 0 else "#3498db" for v in vals]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(labs, vals, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("LIME weight (positive = towards disease)")
    ax.set_title("Local explanation (LIME)")
    fig.tight_layout()
    return fig


def predict_and_explain(*inputs):
    if model is None:
        return "Model not trained yet. Run: python -m src.models.train", None, None

    data = {name: [float(val)] for name, val in zip(feature_names, inputs)}
    X_df = pd.DataFrame(data)
    x_pre = preprocessor.transform(X_df)
    if hasattr(x_pre, "toarray"):
        x_pre = x_pre.toarray()
    x_pre = np.asarray(x_pre)

    shap_values, proba = shap_values_for_input(model, shap_explainer, x_pre)
    lime_list, _ = lime_explanation_for_input(lime_explainer, model.named_steps["clf"], x_pre)

    label = "POSITIVE — disease likely" if proba >= 0.5 else "NEGATIVE — likely benign"
    result = f"{label}\nRisk probability: {proba:.1%}"

    shap_fig = _shap_chart(shap_values, feature_names) if shap_values.size else None
    lime_fig = _lime_chart(lime_list) if lime_list else None
    return result, shap_fig, lime_fig


# Build inputs grouped in 3-column rows
input_components = []

with gr.Blocks(title="DiagnoAI — Breast Cancer Prediction") as demo:
    gr.Markdown("# DiagnoAI — Breast Cancer Prediction\nEnter tumour feature values below to get a risk prediction with explanations.")

    fn = feature_names or [f"feature_{i}" for i in range(30)]
    cols = 3
    for row_start in range(0, len(fn), cols):
        with gr.Row():
            for name in fn[row_start:row_start + cols]:
                input_components.append(gr.Number(label=name, value=0.0))

    predict_btn = gr.Button("Predict", variant="primary")

    with gr.Row():
        prediction_out = gr.Textbox(label="Result", lines=2)

    with gr.Row():
        shap_plot = gr.Plot(label="SHAP — global feature importance")
        lime_plot = gr.Plot(label="LIME — local explanation")

    predict_btn.click(
        fn=predict_and_explain,
        inputs=input_components,
        outputs=[prediction_out, shap_plot, lime_plot],
    )

    gr.Markdown("---\n**Red bars** push towards *positive* (disease). **Blue bars** push towards *negative*.")


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, theme=gr.themes.Soft())
