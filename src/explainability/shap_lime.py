import os
from typing import Any, Dict, List, Tuple
import numpy as np
import joblib

try:
	import shap  # optional
	shap_available = True
except Exception:
	shap_available = False

from lime.lime_tabular import LimeTabularExplainer

ARTIFACTS_DIR = os.path.join("artifacts")


def load_artifacts():
	model = joblib.load(os.path.join(ARTIFACTS_DIR, "model.joblib"))
	preprocessor = joblib.load(os.path.join(ARTIFACTS_DIR, "preprocessor.joblib"))
	feature_names: List[str] = joblib.load(os.path.join(ARTIFACTS_DIR, "feature_names.joblib"))
	background_X = np.load(os.path.join(ARTIFACTS_DIR, "background_X.npy"))
	return model, preprocessor, feature_names, background_X


def build_shap_explainer(model, background_X: np.ndarray):
	if not shap_available:
		return None
	try:
		explainer = shap.Explainer(model.named_steps["clf"], background_X)
	except Exception:
		clf_fallback = model.named_steps["clf"] if hasattr(model, "named_steps") else model
		explainer = shap.KernelExplainer(lambda data: clf_fallback.predict_proba(data)[:, 0], background_X)
	return explainer


def shap_values_for_input(model, explainer, x_pre_row: np.ndarray) -> Tuple[np.ndarray, float]:
	if x_pre_row.ndim == 1:
		x_pre_row = x_pre_row.reshape(1, -1)
	clf = model.named_steps["clf"] if hasattr(model, "named_steps") else model
	# class 0 = malignant (disease), class 1 = benign — return disease probability
	proba = float(clf.predict_proba(x_pre_row)[:, 0][0])
	if not shap_available or explainer is None:
		return np.array([]), proba
	try:
		vals = explainer(x_pre_row)
		shap_values = vals.values if hasattr(vals, "values") else np.array(vals)
	except Exception:
		shap_values = explainer.shap_values(x_pre_row)
	return shap_values, proba


def build_lime_explainer(background_X: np.ndarray, feature_names: List[str]):
	return LimeTabularExplainer(
		training_data=background_X,
		feature_names=feature_names,
		class_names=["benign", "malignant"],
		mode="classification",
		discretize_continuous=True,
	)


def lime_explanation_for_input(lime_explainer, model, x_pre_row: np.ndarray, num_features: int = 8):
	if x_pre_row.ndim == 1:
		x_pre_row = x_pre_row.reshape(1, -1)
	# col 0 = P(benign), col 1 = P(malignant) — LIME explains class 1 (malignant)
	predict_fn = lambda data: np.column_stack(
		[
			model.predict_proba(data)[:, 1],
			model.predict_proba(data)[:, 0],
		]
	)
	exp = lime_explainer.explain_instance(x_pre_row[0], predict_fn, num_features=num_features)
	return exp.as_list(), exp.score
