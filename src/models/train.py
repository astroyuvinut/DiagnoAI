import os
import json
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
try:
	from xgboost import XGBClassifier  # optional
	xgb_available = True
except Exception:
	xgb_available = False
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV

from src.utils.preprocess import ensure_artifacts_dir, load_dataset, build_preprocessor, split_data, sample_background_X, ARTIFACTS_DIR


def evaluate_classifier(y_true: np.ndarray, y_proba: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
	return {
		"accuracy": float(accuracy_score(y_true, y_pred)),
		"precision": float(precision_score(y_true, y_pred)),
		"recall": float(recall_score(y_true, y_pred)),
		"f1": float(f1_score(y_true, y_pred)),
		"roc_auc": float(roc_auc_score(y_true, y_proba)),
	}


def plot_and_save_roc(y_true: np.ndarray, y_proba: np.ndarray, model_name: str, out_path: str) -> None:
	fpr, tpr, _ = roc_curve(y_true, y_proba)
	plt.figure(figsize=(6, 5))
	plt.plot(fpr, tpr, label=f"{model_name} (AUC={roc_auc_score(y_true, y_proba):.3f})")
	plt.plot([0, 1], [0, 1], "k--", label="Chance")
	plt.xlabel("False Positive Rate")
	plt.ylabel("True Positive Rate")
	plt.title("ROC Curve")
	plt.legend(loc="lower right")
	plt.tight_layout()
	plt.savefig(out_path, dpi=150)
	plt.close()


def main() -> Tuple[str, Dict[str, Any]]:
	ensure_artifacts_dir()
	X_df, y, feature_names = load_dataset()
	preprocessor = build_preprocessor(feature_names)
	X_train, X_test, y_train, y_test = split_data(X_df, y)

	# Baseline models
	models = {
		"log_reg": LogisticRegression(max_iter=2000, class_weight="balanced", n_jobs=None),
		"rf": RandomForestClassifier(class_weight="balanced", random_state=42),
	}
	if xgb_available:
		models["xgb"] = XGBClassifier(objective="binary:logistic", eval_metric="logloss", tree_method="hist", random_state=42)

	# Simple grids focusing on recall-friendly settings
	grids = {
		"log_reg": {
			"C": [0.1, 1.0, 3.0],
		},
		"rf": {
			"n_estimators": [200, 400],
			"max_depth": [None, 6, 12],
			"min_samples_split": [2, 5],
		},
	}
	if xgb_available:
		grids["xgb"] = {
			"n_estimators": [200, 400],
			"max_depth": [3, 5],
			"learning_rate": [0.05, 0.1],
			"subsample": [0.8, 1.0],
			"colsample_bytree": [0.8, 1.0],
		}

	best_model_name = None
	best_model = None
	best_metrics: Dict[str, float] = {"roc_auc": -1.0}

	for name, base_model in models.items():
		print(f"Tuning {name}...")
		# Use pipeline: preprocessor + model
		from sklearn.pipeline import Pipeline as SkPipeline
		pipe = SkPipeline(steps=[("pre", preprocessor), ("clf", base_model)])
		param_grid = {f"clf__{k}": v for k, v in grids[name].items()}
		search = GridSearchCV(
			pipe,
			param_grid=param_grid,
			scoring="recall",
			cv=5,
			n_jobs=-1,
			verbose=1,
		)
		search.fit(X_train, y_train)
		y_proba = search.predict_proba(X_test)[:, 1]
		y_pred = (y_proba >= 0.5).astype(int)
		metrics = evaluate_classifier(y_test.values, y_proba, y_pred)
		print(name, metrics)

		roc_path = os.path.join(ARTIFACTS_DIR, f"roc_{name}.png")
		plot_and_save_roc(y_test.values, y_proba, name, roc_path)

		if metrics["roc_auc"] > best_metrics.get("roc_auc", -1):
			best_metrics = metrics
			best_model_name = name
			best_model = search.best_estimator_

	# Persist artifacts
	assert best_model is not None and best_model_name is not None
	joblib.dump(best_model, os.path.join(ARTIFACTS_DIR, "model.joblib"))
	joblib.dump(feature_names, os.path.join(ARTIFACTS_DIR, "feature_names.joblib"))

	# Save small background (post-preprocessing) for SHAP/LIME
	# Fit preprocessor alone and transform training set
	from sklearn.base import clone
	pre_only = clone(preprocessor)
	pre_only.fit(X_train, y_train)
	x_train_pre = pre_only.transform(X_train)
	if hasattr(x_train_pre, "toarray"):
		x_train_pre = x_train_pre.toarray()
	bg = sample_background_X(np.asarray(x_train_pre), max_samples=200)
	np.save(os.path.join(ARTIFACTS_DIR, "background_X.npy"), bg)
	joblib.dump(pre_only, os.path.join(ARTIFACTS_DIR, "preprocessor.joblib"))

	with open(os.path.join(ARTIFACTS_DIR, "metrics.json"), "w", encoding="utf-8") as f:
		json.dump({"best_model": best_model_name, "metrics": best_metrics}, f, indent=2)

	print(f"Best model: {best_model_name}")
	print(json.dumps(best_metrics, indent=2))
	return best_model_name, best_metrics


if __name__ == "__main__":
	main()
