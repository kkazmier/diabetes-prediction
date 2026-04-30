from pathlib import Path

import pandas as pd

from config import config
from src.models import get_models
from src.training import run_nested_cv
from src.utils import print_results_summary, save_results_plot, save_results_table


def main() -> None:
    print("=== Diabetes Classification with Optuna ===\n")

    data_path = Path("data/diabetes.csv")
    df = pd.read_csv(data_path)

    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    results = []
    models = get_models()

    for model_name in models.keys():
        print(f"\n{'=' * 60}")
        print(f"Training {model_name}")
        print(f"{'=' * 60}")

        result = run_nested_cv(X, y, model_name)
        results.append(result)

    results_df = pd.DataFrame(results).sort_values("roc_auc_mean", ascending=False)

    print_results_summary(results_df)

    results_to_save = results_df.drop(columns=["fold_details"], errors="ignore")

    save_results_table(results_to_save, "results/final_results.csv")
    save_results_plot(results_to_save, "results/plots/model_comparison.png")

    print("\nWyniki zapisano do katalogu 'results/'.")


if __name__ == "__main__":
    main()
