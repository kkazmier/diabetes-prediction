from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import optuna
import pandas as pd
from optuna.samplers import TPESampler
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from config import config
from src.models import get_models, get_param_space
from src.preprocessing import create_preprocessing_pipeline
from src.utils import save_fold_details_json, save_fold_details_csv

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _suggest_params(trial: optuna.Trial, model_name: str) -> dict[str, Any]:
    """
    Generuje hiperparametry dla danego modelu na podstawie przestrzeni
    zdefiniowanej w models.py.
    """
    param_space = get_param_space(model_name)
    params: dict[str, Any] = {}

    for param_name, space in param_space.items():
        if isinstance(space, tuple):
            if len(space) == 3 and space[2] == "log-uniform":
                params[param_name] = trial.suggest_float(
                    param_name, space[0], space[1], log=True
                )
            elif len(space) == 2:
                low, high = space
                if isinstance(low, int) and isinstance(high, int):
                    params[param_name] = trial.suggest_int(param_name, low, high)
                else:
                    params[param_name] = trial.suggest_float(param_name, low, high)
            else:
                raise ValueError(
                    f"Nieobsługiwany format przestrzeni dla parametru "
                    f"'{param_name}': {space}"
                )

        elif isinstance(space, list):
            params[param_name] = trial.suggest_categorical(param_name, space)

        else:
            raise ValueError(
                f"Nieobsługiwany typ przestrzeni dla parametru "
                f"'{param_name}': {space}"
            )

    return params


def _predict_scores(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """
    Zwraca score/probability do ROC-AUC.
    Priorytet:
    1. predict_proba[:, 1]
    2. decision_function
    """
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]

    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        scores = np.asarray(scores, dtype=float)

        min_val = scores.min()
        max_val = scores.max()

        if max_val > min_val:
            scores = (scores - min_val) / (max_val - min_val)

        return scores

    raise AttributeError(
        "Model nie posiada ani 'predict_proba', ani 'decision_function'."
    )


def _build_full_pipeline(
    model_name: str,
    model_params: dict[str, Any],
    impute_strategy: str,
    n_features: int,
) -> Pipeline:
    """
    Buduje pełny pipeline:
    preprocessing + classifier
    """
    base_model = clone(get_models()[model_name])
    model = base_model.set_params(**model_params)

    pipeline = Pipeline(
        steps=[
            (
                "preprocess",
                create_preprocessing_pipeline(
                    impute_strategy=impute_strategy,
                    n_features=n_features,
                ),
            ),
            ("classifier", model),
        ]
    )
    return pipeline


def _objective(
    trial: optuna.Trial,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_name: str,
    impute_strategy: str,
) -> float:
    """
    Funkcja celu dla Optuna.
    Optymalizuje średni ROC-AUC w inner CV.
    """
    model_params = _suggest_params(trial, model_name)

    inner_cv = StratifiedKFold(
        n_splits=config.inner_cv,
        shuffle=True,
        random_state=config.random_state,
    )

    scores: list[float] = []

    for train_idx, val_idx in inner_cv.split(X_train, y_train):
        X_tr = X_train.iloc[train_idx]
        X_val = X_train.iloc[val_idx]
        y_tr = y_train.iloc[train_idx]
        y_val = y_train.iloc[val_idx]

        pipeline = _build_full_pipeline(
            model_name=model_name,
            model_params=model_params,
            impute_strategy=impute_strategy,
            n_features=config.n_features,
        )

        pipeline.fit(X_tr, y_tr)
        y_scores = _predict_scores(pipeline, X_val)
        fold_score = roc_auc_score(y_val, y_scores)
        scores.append(fold_score)

    mean_score = float(np.mean(scores))
    return mean_score


def run_nested_cv(X: pd.DataFrame, y: pd.Series, model_name: str) -> dict[str, Any]:
    """
    Uruchamia nested cross-validation dla jednego modelu:
    - outer CV: uczciwa ocena
    - inner CV + Optuna: tuning hiperparametrów

    Zwraca słownik z podsumowaniem wyników.
    """
    outer_cv = StratifiedKFold(
        n_splits=config.outer_cv,
        shuffle=True,
        random_state=config.random_state,
    )

    outer_scores: list[float] = []
    chosen_imputations: list[str] = []
    fold_details: list[dict[str, Any]] = []
    fitted_pipelines: list[Pipeline] = []

    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), start=1):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        best_inner_score = -np.inf
        best_impute_strategy = None
        best_params: dict[str, Any] | None = None

        print(f"\n[Model: {model_name}] Outer fold {fold_idx}/{config.outer_cv}")

        for strategy in config.impute_strategies:
            print(f"  -> Optuna | imputacja: {strategy}")

            study = optuna.create_study(
                direction="maximize",
                sampler=TPESampler(seed=config.random_state),
            )

            study.optimize(
                lambda trial: _objective(
                    trial=trial,
                    X_train=X_train,
                    y_train=y_train,
                    model_name=model_name,
                    impute_strategy=strategy,
                ),
                n_trials=config.n_trials,
                show_progress_bar=False,
            )

            if study.best_value > best_inner_score:
                best_inner_score = float(study.best_value)
                best_impute_strategy = strategy
                best_params = study.best_params

        if best_impute_strategy is None or best_params is None:
            raise RuntimeError("Nie udało się znaleźć najlepszej konfiguracji.")

        best_pipeline = _build_full_pipeline(
            model_name=model_name,
            model_params=best_params,
            impute_strategy=best_impute_strategy,
            n_features=config.n_features,
        )

        best_pipeline.fit(X_train, y_train)
        y_scores = _predict_scores(best_pipeline, X_test)
        outer_score = float(roc_auc_score(y_test, y_scores))

        outer_scores.append(outer_score)
        chosen_imputations.append(best_impute_strategy)
        fitted_pipelines.append(best_pipeline)

        fold_details.append(
            {
                "fold": fold_idx,
                "outer_roc_auc": outer_score,
                "best_inner_roc_auc": best_inner_score,
                "best_impute_strategy": best_impute_strategy,
                "best_params": best_params,
            }
        )

        print(
            f"  Fold result | inner ROC-AUC: {best_inner_score:.4f} "
            f"| test ROC-AUC: {outer_score:.4f} "
            f"| imputacja: {best_impute_strategy}"
        )

    mean_score = float(np.mean(outer_scores))
    std_score = float(np.std(outer_scores))

    best_fold_index = int(np.argmax(outer_scores))
    best_pipeline = fitted_pipelines[best_fold_index]

    model_dir = Path("results/best_models")
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / f"{model_name}_best.joblib"
    joblib.dump(best_pipeline, model_path)

    best_imputation_overall = max(
        set(chosen_imputations),
        key=chosen_imputations.count
    )

    json_path = save_fold_details_json(fold_details, model_name)
    csv_path = save_fold_details_csv(fold_details, model_name)

    return {
        "model": model_name,
        "roc_auc_mean": mean_score,
        "roc_auc_std": std_score,
        "best_impute_strategy": best_imputation_overall,
        "best_model_path": str(model_path),
        "fold_details_json_path": str(json_path),
        "fold_details_csv_path": str(csv_path),
        "fold_details": fold_details,
    }


if __name__ == "__main__":
    df = pd.read_csv("data/diabetes.csv")
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    result = run_nested_cv(X, y, "LogisticRegression")
    print(result)


