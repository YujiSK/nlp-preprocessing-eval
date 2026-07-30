"""全実験共通の4モデル定義（docs/execution_plan.md 第2章）。"""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

from ..utils import RANDOM_STATE

MODEL_ORDER = ["logistic_regression", "linear_svc", "random_forest", "knn"]

# k-NNには class_weight パラメータが存在しない（計画書 3章・実験C）。
SUPPORTS_CLASS_WEIGHT = {"logistic_regression", "linear_svc", "random_forest"}

_DEFAULTS = {
    "logistic_regression": dict(random_state=RANDOM_STATE, max_iter=2000),
    "linear_svc": dict(random_state=RANDOM_STATE, max_iter=5000),
    "random_forest": dict(random_state=RANDOM_STATE),
    "knn": dict(),
}

_CONSTRUCTORS = {
    "logistic_regression": LogisticRegression,
    "linear_svc": LinearSVC,
    "random_forest": RandomForestClassifier,
    "knn": KNeighborsClassifier,
}


def build_model(model_name: str, **overrides):
    """モデル名から新規インスタンスを生成する。

    overridesでハイパーパラメータ（例: C, n_neighbors, max_depth, class_weight）を上書きする。
    """
    if model_name not in _CONSTRUCTORS:
        raise ValueError(f"unknown model_name: {model_name}")

    params = dict(_DEFAULTS[model_name])
    if "class_weight" in overrides and model_name not in SUPPORTS_CLASS_WEIGHT:
        raise ValueError(f"{model_name} does not support class_weight")
    params.update(overrides)
    return _CONSTRUCTORS[model_name](**params)
