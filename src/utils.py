import json
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def ensure_dir(path: str | Path) -> Path:
    """
    Tworzy katalog, jeśli nie istnieje, i zwraca Path.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def print_results_summary(results_df: pd.DataFrame) -> None:
    """
    Czytelne wypisanie podsumowania wyników modeli.
    """
    if results_df.empty:
        print("Brak wyników do wyświetlenia.")
        return

    print("\n" + "=" * 70)
    print("PODSUMOWANIE WYNIKÓW")
    print("=" * 70)
    print(results_df.to_string(index=False))
    print("=" * 70)

    best_row = results_df.iloc[0]
    print("\nNajlepszy model:")
    print(f"  Model: {best_row['model']}")
    print(f"  Średni ROC-AUC: {best_row['roc_auc_mean']:.4f}")
    print(f"  STD ROC-AUC: {best_row['roc_auc_std']:.4f}")
    print(f"  Najlepsza imputacja: {best_row['best_impute_strategy']}")
    print(f"  Ścieżka modelu: {best_row['best_model_path']}")

    if "fold_details_json_path" in best_row:
        print(f"  Fold details JSON: {best_row['fold_details_json_path']}")
    if "fold_details_csv_path" in best_row:
        print(f"  Fold details CSV: {best_row['fold_details_csv_path']}")


def save_results_plot(
    results_df: pd.DataFrame,
    save_path: str | Path = "results/plots/model_comparison.png",
    figsize: tuple[int, int] = (10, 6),
    dpi: int = 150,
) -> Path:
    """
    Zapisuje wykres porównujący modele po ROC-AUC.
    """
    if results_df.empty:
        raise ValueError("results_df jest pusty.")

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plot_df = results_df.copy()
    plot_df = plot_df.sort_values("roc_auc_mean", ascending=False)

    plt.figure(figsize=figsize)
    sns.barplot(
        data=plot_df,
        x="roc_auc_mean",
        y="model",
        hue="model",
        dodge=False,
        palette="viridis",
        legend=False,
    )

    for i, row in enumerate(plot_df.itertuples(index=False)):
        plt.errorbar(
            x=row.roc_auc_mean,
            y=i,
            xerr=row.roc_auc_std,
            fmt="none",
            ecolor="black",
            capsize=4,
        )

    plt.title("Porównanie modeli (ROC-AUC)")
    plt.xlabel("Średni ROC-AUC")
    plt.ylabel("Model")
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi)
    plt.close()

    return save_path


def save_results_table(
    results_df: pd.DataFrame,
    save_path: str | Path = "results/final_results.csv"
) -> Path:
    """
    Zapisuje tabelę wyników do CSV.
    """
    if results_df.empty:
        raise ValueError("results_df jest pusty.")

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(save_path, index=False)
    return save_path


def load_csv(path: str | Path) -> pd.DataFrame:
    """
    Wczytuje plik CSV i zwraca DataFrame.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku: {path}")
    return pd.read_csv(path)


def basic_eda_report(
    df: pd.DataFrame,
    target_col: Optional[str] = None,
    save_dir: str | Path = "results/eda"
) -> None:
    """
    Prosty raport EDA:
    - shape
    - info
    - describe
    - rozkład klas
    - heatmapa korelacji
    - histogramy

    Wyniki graficzne zapisuje do plików PNG.
    """
    save_dir = ensure_dir(save_dir)

    print("\n" + "=" * 70)
    print("RAPORT EDA")
    print("=" * 70)
    print(f"Shape: {df.shape}")
    print("\nInfo:")
    print(df.info())
    print("\nStatystyki opisowe:")
    print(df.describe())

    if target_col and target_col in df.columns:
        print(f"\nRozkład klas dla '{target_col}':")
        print(df[target_col].value_counts())

        plt.figure(figsize=(6, 4))
        sns.countplot(data=df, x=target_col)
        plt.title(f"Rozkład klas: {target_col}")
        plt.tight_layout()
        plt.savefig(save_dir / "class_distribution.png", dpi=150)
        plt.close()

    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Macierz korelacji")
    plt.tight_layout()
    plt.savefig(save_dir / "correlation_heatmap.png", dpi=150)
    plt.close()

    df.hist(figsize=(14, 10), bins=20, grid=False, color="#4C72B0")
    plt.tight_layout()
    plt.savefig(save_dir / "histograms.png", dpi=150)
    plt.close()

    print(f"\nRaport EDA zapisano w: {save_dir}")


def save_fold_details_json(
    fold_details: list[dict[str, Any]],
    model_name: str,
    save_dir: str | Path = "results/folds"
) -> Path:
    """
    Zapisuje szczegóły foldów do pliku JSON.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    save_path = save_dir / f"{model_name}_fold_details.json"

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(fold_details, f, indent=2, ensure_ascii=False)

    return save_path


def save_fold_details_csv(
    fold_details: list[dict[str, Any]],
    model_name: str,
    save_dir: str | Path = "results/folds"
) -> Path:
    """
    Zapisuje szczegóły foldów do CSV.

    Kolumna 'best_params' zostaje zamieniona na tekst JSON,
    żeby CSV było łatwe do otwarcia.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    save_path = save_dir / f"{model_name}_fold_details.csv"

    df = pd.DataFrame(fold_details).copy()

    if "best_params" in df.columns:
        df["best_params"] = df["best_params"].apply(
            lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, dict) else x
        )

    df.to_csv(save_path, index=False, encoding="utf-8")
    return save_path
