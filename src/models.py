from typing import Any

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


def get_models() -> dict[str, Any]:
    """
    Zwraca bazowe instancje modeli.

    Modele są później klonowane i dostrajane w training.py.
    """
    return {
        "LogisticRegression": LogisticRegression(
            max_iter=2000,
            random_state=42,
        ),
        "SVM": SVC(
            probability=True,
            random_state=42,
        ),
        "RandomForest": RandomForestClassifier(
            random_state=42,
        ),
        "GradientBoosting": GradientBoostingClassifier(
            random_state=42,
        ),
    }


def get_param_space(model_name: str) -> dict[str, Any]:
    """
    Zwraca przestrzeń hiperparametrów dla Optuna.

    Obsługiwane formaty:
    - (low, high)                  -> int lub float
    - (low, high, "log-uniform")   -> float log-scale
    - [v1, v2, ...]                -> categorical
    """
    spaces: dict[str, dict[str, Any]] = {
        "LogisticRegression": {
            "C": (1e-4, 10.0, "log-uniform"),
            "solver": ["lbfgs", "liblinear"],
        },
        "SVM": {
            "C": (1e-3, 50.0, "log-uniform"),
            "kernel": ["rbf", "linear"],
        },
        "RandomForest": {
            "n_estimators": (50, 300),
            "max_depth": (3, 20),
            "min_samples_split": (2, 10),
            "min_samples_leaf": (1, 5),
        },
        "GradientBoosting": {
            "n_estimators": (50, 300),
            "learning_rate": (0.01, 0.3, "log-uniform"),
            "max_depth": (2, 8),
            "subsample": (0.6, 1.0),
        },
    }

    if model_name not in spaces:
        available = ", ".join(spaces.keys())
        raise ValueError(
            f"Nieznany model: '{model_name}'. Dostępne modele: {available}"
        )

    return spaces[model_name]
