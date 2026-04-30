from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import optuna
import pandas as pd
from optuna.samplers import TPESampler
from sklearn.base import clone
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from config import config
from src.models import get_models, get_param_space
from src.preprocessing import create_preprocessing_pipeline
from src.utils import save_fold_details_json, save_fold_details_csv

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _suggest_params(trial: optuna.Trial, model_name: str) -> dict[str, Any]:
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


def _predict_classes(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    return model.predict(X)


def _build_full_pipeline(
    model_name: str,
    model_params: dict[str, Any],
    impute_strategy: str,
    n_features: int,
) -> Pipeline:
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
        scores.append(roc_auc_score(y_val, y_scores))

    return float(np.mean(scores))


def run_nested_cv(X: pd.DataFrame, y: pd.Series, model_name: str) -> dict[str, Any]:
    outer_cv = StratifiedKFold(
        n_splits=config.outer_cv,
        shuffle=True,
        random_state=config.random_state,
    )

    outer_roc_scores = []
    outer_acc_scores = []
    outer_f1_scores = []

    chosen_imputations = []
    fold_details = []
    fitted_pipelines = []

    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), start=1):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        best_inner_score = -np.inf
        best_impute_strategy = None
        best_params = None

        print(f"\n[Model: {model_name}] Outer fold {fold_idx}/{config.outer_cv}")

        for strategy in config.impute_strategies:
            print(f"  -> Optuna | imputacja: {strategy}")

            study = optuna.create_study(
                direction="maximize",
                sampler=TPESampler(seed=config.random_state),
            )

            study.optimize(
                lambda trial: _objective(
                    trial,
                    X_train,
                    y_train,
                    model_name,
                    strategy,
                ),
                n_trials=config.n_trials,
                show_progress_bar=False,
            )

            if study.best_value > best_inner_score:
                best_inner_score = float(study.best_value)
                best_impute_strategy = strategy
                best_params = study.best_params

        best_pipeline = _build_full_pipeline(
            model_name=model_name,
            model_params=best_params,
            impute_strategy=best_impute_strategy,
            n_features=config.n_features,
        )

        best_pipeline.fit(X_train, y_train)

        y_scores = _predict_scores(best_pipeline, X_test)
        y_pred = _predict_classes(best_pipeline, X_test)

        outer_roc = float(roc_auc_score(y_test, y_scores))
        outer_acc = float(accuracy_score(y_test, y_pred))
        outer_f1 = float(f1_score(y_test, y_pred))

        outer_roc_scores.append(outer_roc)
        outer_acc_scores.append(outer_acc)
        outer_f1_scores.append(outer_f1)

        chosen_imputations.append(best_impute_strategy)
        fitted_pipelines.append(best_pipeline)

        fold_details.append(
            {
                "fold": fold_idx,
                "accuracy": outer_acc,
                "f1_score": outer_f1,
                "outer_roc_auc": outer_roc,
                "best_inner_roc_auc": best_inner_score,
                "best_impute_strategy": best_impute_strategy,
                "best_params": best_params,
            }
        )

        print(
            f"  Fold result | "
            f"ACC: {outer_acc:.4f} | "
            f"F1: {outer_f1:.4f} | "
            f"ROC-AUC: {outer_roc:.4f} | "
            f"imputacja: {best_impute_strategy}"
        )

    roc_mean = float(np.mean(outer_roc_scores))
    roc_std = float(np.std(outer_roc_scores))

    acc_mean = float(np.mean(outer_acc_scores))
    acc_std = float(np.std(outer_acc_scores))

    f1_mean = float(np.mean(outer_f1_scores))
    f1_std = float(np.std(outer_f1_scores))

    best_fold_index = int(np.argmax(outer_roc_scores))
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
        "accuracy_mean": acc_mean,
        "accuracy_std": acc_std,
        "f1_mean": f1_mean,
        "f1_std": f1_std,
        "roc_auc_mean": roc_mean,
        "roc_auc_std": roc_std,
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
