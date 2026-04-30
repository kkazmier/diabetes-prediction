from dataclasses import dataclass
from typing import Any

@dataclass
class Config:
    random_state: int = 42
    outer_cv: int = 3
    inner_cv: int = 2
    n_trials: int = 5
    n_features: int = 6
    # n_jobs = 2
    impute_strategies: list[str] = None
    test_size: float = 0.2

    def __post_init__(self):
        if self.impute_strategies is None:
            self.impute_strategies = ["mean", "median", "knn"]


config = Config()
