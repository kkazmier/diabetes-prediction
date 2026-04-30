import numpy as np
import pandas as pd
from typing import List

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class ReplaceInvalidValues(BaseEstimator, TransformerMixin):
    """Zamienia biologicznie niemożliwe wartości 0 na NaN."""

    def __init__(self, columns: List[str] | None = None):
        self.columns = columns or ["Glucose", "BloodPressure", "BMI", "Insulin"]

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X[self.columns] = X[self.columns].replace(0, np.nan)
        return X


def create_preprocessing_pipeline(
    impute_strategy: str = "median",
    n_features: int = 6,
    n_neighbors: int = 5,
) -> Pipeline:
    """Tworzy preprocessing pipeline."""

    if impute_strategy == "knn":
        imputer = KNNImputer(n_neighbors=n_neighbors)
    else:
        imputer = SimpleImputer(strategy=impute_strategy)

    numeric_features = [
        "Pregnancies",
        "Glucose",
        "BloodPressure",
        "BMI",
        "DiabetesPedigreeFunction",
        "Age",
        "Insulin",
        "SkinThickness",
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", imputer),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            )
        ],
        remainder="drop",
    )

    pipeline = Pipeline(
        steps=[
            ("replace_invalid", ReplaceInvalidValues()),
            ("preprocessor", preprocessor),
            ("feature_selection", SelectKBest(score_func=mutual_info_classif, k=n_features)),
        ]
    )

    return pipeline
