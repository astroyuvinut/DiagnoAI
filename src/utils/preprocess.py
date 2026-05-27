import os
from typing import Tuple, List
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

ARTIFACTS_DIR = os.path.join("artifacts")


def ensure_artifacts_dir() -> None:
	os.makedirs(ARTIFACTS_DIR, exist_ok=True)


def load_dataset() -> Tuple[pd.DataFrame, pd.Series, List[str]]:
	data = load_breast_cancer(as_frame=True)
	X: pd.DataFrame = data.frame.drop(columns=["target"]).copy()
	y: pd.Series = data.target.copy()
	feature_names = list(X.columns)
	return X, y, feature_names


def build_preprocessor(feature_names: List[str]) -> ColumnTransformer:
	numeric_features = feature_names
	numeric_transformer = Pipeline(steps=[
		("imputer", SimpleImputer(strategy="median")),
		("scaler", StandardScaler(with_mean=True, with_std=True)),
	])
	preprocessor = ColumnTransformer(
		transformers=[
			("num", numeric_transformer, numeric_features),
		],
		remainder="drop",
	)
	return preprocessor


def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42):
	return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def sample_background_X(X: np.ndarray, max_samples: int = 200, random_state: int = 42) -> np.ndarray:
	if X.shape[0] <= max_samples:
		return X
	rng = np.random.default_rng(random_state)
	idx = rng.choice(X.shape[0], size=max_samples, replace=False)
	return X[idx]
