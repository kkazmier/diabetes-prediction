# main_pipeline.py
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    make_scorer, accuracy_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    RocCurveDisplay, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')  # Zapis do pliku zamiast wyświetlania
import warnings

warnings.filterwarnings('ignore')

# ====================== KONFIGURACJA ======================
DATA_PATH = 'data/diabetes.csv'
RESULTS_DIR = 'results/'
import os

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR + 'plots/', exist_ok=True)


# ====================== WCZYTANIE DANYCH ======================
def load_data(path):
    """
    Wczytuje dane i zastępuje wartości 0 na NaN w kolumnach biologicznych,
    gdzie zero jest fizycznie niemożliwe (np. BMI=0, Glucose=0).
    """
    df = pd.read_csv(path)
    cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    df[cols_with_zeros] = df[cols_with_zeros].replace(0, np.nan)

    print("=== INFO O ZBIORZE DANYCH ===")
    print(f"Rozmiar: {df.shape}")
    print(f"\nBrakujące wartości:\n{df.isnull().sum()}")
    print(f"\nRozkład klas:\n{df['Outcome'].value_counts()}")
    print(f"Balans klas: {df['Outcome'].value_counts(normalize=True).round(3).to_dict()}")
    print("=" * 40)
    return df


# ====================== BUDOWANIE PIPELINE ======================
def build_pipeline(imputer, selector, model):
    """
    Buduje potok przetwarzania danych (Pipeline), który:
    1. Imputuje brakujące wartości (imputer)
    2. Standaryzuje cechy (StandardScaler)
    3. Wybiera najważniejsze cechy (selector)
    4. Klasyfikuje (model)

    Użycie Pipeline gwarantuje brak data leakage podczas walidacji krzyżowej -
    każdy krok jest fitowany TYLKO na danych treningowych danego foldu.
    """
    return Pipeline([
        ('imputer', imputer),
        ('scaler', StandardScaler()),
        ('selector', selector),
        ('classifier', model)
    ])


# ====================== SCORER DLA ROC-AUC ======================
def safe_roc_auc_scorer(estimator, X, y):
    """
    Własna funkcja scorera dla ROC-AUC.
    Używa predict_proba jeśli dostępne, w przeciwnym razie decision_function.
    Rozwiązuje problem NaN przy SVC i innych modelach.
    """
    if hasattr(estimator, "predict_proba"):
        y_score = estimator.predict_proba(X)[:, 1]
    elif hasattr(estimator, "decision_function"):
        y_score = estimator.decision_function(X)
    else:
        raise ValueError("Model nie obsługuje predict_proba ani decision_function")
    return roc_auc_score(y, y_score)


# ====================== EKSPERYMENTY ======================
def run_experiments(X, y):
    """
    Automatyzuje eksperymenty dla wszystkich kombinacji:
    - 5 modeli x 2 metody imputacji x 2 metody selekcji cech = 20 konfiguracji

    Używa StratifiedKFold (k=5), który zachowuje proporcje klas w każdym foldzie.
    Zwraca DataFrame z wynikami oraz odchyleniami standardowymi (analiza stabilności).
    """

    # --- Modele ---
    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
        'RandomForest': RandomForestClassifier(n_estimators=200, random_state=42),
        'XGBoost': XGBClassifier(
            n_estimators=200, learning_rate=0.1,
            random_state=42, eval_metric='logloss',
            verbosity=0
        ),
        'SVM': SVC(probability=True, kernel='rbf', random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5)
    }

    # --- Dwie metody imputacji ---
    imputers = {
        'Median': SimpleImputer(strategy='median'),
        'KNN_Imputer': KNNImputer(n_neighbors=5)
    }

    # --- Dwie metody selekcji cech ---
    selectors = {
        'SelectKBest_ANOVA': SelectKBest(score_func=f_classif, k=6),
        'SelectFromModel_RF': SelectFromModel(
            estimator=RandomForestClassifier(n_estimators=100, random_state=42),
            max_features=6,
            threshold=-np.inf
        )
    }

    # --- Metryki (NAPRAWIONY scorer dla ROC-AUC) ---
    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'f1': make_scorer(f1_score, zero_division=0),
        'roc_auc': safe_roc_auc_scorer  # <-- własna funkcja zamiast make_scorer
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = []

    print("\n=== ROZPOCZYNAM EKSPERYMENTY ===\n")
    print(f"{'Model':<22} {'Imputer':<14} {'Selector':<28} {'F1':>8} {'AUC':>8} {'ACC':>8}")
    print("-" * 90)

    for model_name, model in models.items():
        for imp_name, imputer in imputers.items():
            for sel_name, selector in selectors.items():
                pipeline = build_pipeline(
                    imputer=imputer,
                    selector=selector,
                    model=model
                )

                scores = cross_validate(
                    pipeline, X, y,
                    cv=skf,
                    scoring=scoring,
                    return_train_score=False,
                    n_jobs=-1
                )

                acc_mean = scores['test_accuracy'].mean()
                acc_std = scores['test_accuracy'].std()
                f1_mean = scores['test_f1'].mean()
                f1_std = scores['test_f1'].std()
                auc_mean = scores['test_roc_auc'].mean()
                auc_std = scores['test_roc_auc'].std()

                results.append({
                    'Model': model_name,
                    'Imputer': imp_name,
                    'Selector': sel_name,
                    'Accuracy': round(acc_mean, 4),
                    'Accuracy_std': round(acc_std, 4),
                    'F1': round(f1_mean, 4),
                    'F1_std': round(f1_std, 4),
                    'ROC_AUC': round(auc_mean, 4),
                    'ROC_AUC_std': round(auc_std, 4),
                })

                print(f"{model_name:<22} {imp_name:<14} {sel_name:<28} "
                      f"{f1_mean:>7.4f} {auc_mean:>8.4f} {acc_mean:>8.4f}")

    return pd.DataFrame(results)


# ====================== WYKRESY ======================
def plot_comparison(results_df):
    """Wykres słupkowy porównujący F1 i ROC-AUC dla wszystkich konfiguracji."""
    results_df['Config'] = (results_df['Model'] + '\n' +
                            results_df['Imputer'] + '\n' +
                            results_df['Selector'].str[:10])

    fig, axes = plt.subplots(2, 1, figsize=(18, 12))

    # F1-Score
    axes[0].bar(range(len(results_df)), results_df['F1'],
                yerr=results_df['F1_std'], capsize=3,
                color='steelblue', alpha=0.8)
    axes[0].set_xticks(range(len(results_df)))
    axes[0].set_xticklabels(results_df['Config'], fontsize=7, rotation=45, ha='right')
    axes[0].set_title('F1-Score (mean ± std) dla wszystkich konfiguracji', fontsize=13)
    axes[0].set_ylabel('F1-Score')
    axes[0].set_ylim(0.4, 0.9)
    axes[0].axhline(results_df['F1'].mean(), color='red', linestyle='--',
                    label=f'Średnia F1: {results_df["F1"].mean():.4f}')
    axes[0].legend()

    # ROC-AUC
    axes[1].bar(range(len(results_df)), results_df['ROC_AUC'],
                yerr=results_df['ROC_AUC_std'], capsize=3,
                color='darkorange', alpha=0.8)
    axes[1].set_xticks(range(len(results_df)))
    axes[1].set_xticklabels(results_df['Config'], fontsize=7, rotation=45, ha='right')
    axes[1].set_title('ROC-AUC (mean ± std) dla wszystkich konfiguracji', fontsize=13)
    axes[1].set_ylabel('ROC-AUC')
    axes[1].set_ylim(0.5, 1.0)
    axes[1].axhline(results_df['ROC_AUC'].mean(), color='red', linestyle='--',
                    label=f'Średnia AUC: {results_df["ROC_AUC"].mean():.4f}')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(RESULTS_DIR + 'plots/comparison_all.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Zapisano: results/plots/comparison_all.png")


def plot_calibration(X, y):
    """
    Krzywe kalibracji dla najlepszych modeli.
    Sprawdza czy przewidywane prawdopodobieństwa odpowiadają rzeczywistym częstościom.
    Model idealnie skalibrowany leży na przekątnej.
    Używa calibration_curve zamiast CalibrationDisplay (kompatybilność ze starszym sklearn).
    """
    from sklearn.model_selection import train_test_split
    from sklearn.calibration import calibration_curve

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    calibration_models = {
        'LogisticRegression': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(max_iter=1000, random_state=42))
        ]),
        'RandomForest': Pipeline([
            ('imputer', KNNImputer(n_neighbors=5)),
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(n_estimators=200, random_state=42))
        ]),
        'XGBoost': Pipeline([
            ('imputer', KNNImputer(n_neighbors=5)),
            ('scaler', StandardScaler()),
            ('classifier', XGBClassifier(
                n_estimators=200, random_state=42,
                eval_metric='logloss', verbosity=0
            ))
        ]),
        'SVM': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('classifier', SVC(probability=True, random_state=42))
        ])
    }

    fig, ax = plt.subplots(figsize=(8, 7))

    # Linia idealnej kalibracji
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray',
            label='Idealna kalibracja', linewidth=2)

    colors = ['steelblue', 'darkorange', 'green', 'red']

    for (name, pipe), color in zip(calibration_models.items(), colors):
        pipe.fit(X_train, y_train)
        y_prob = pipe.predict_proba(X_test)[:, 1]

        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_test, y_prob, n_bins=10
        )

        ax.plot(mean_predicted_value, fraction_of_positives,
                marker='o', label=name, color=color, linewidth=2)

    ax.set_xlabel('Średnie przewidywane prawdopodobieństwo', fontsize=11)
    ax.set_ylabel('Frakcja rzeczywistych pozytywów', fontsize=11)
    ax.set_title('Krzywe kalibracji modeli', fontsize=13, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR + 'plots/calibration_curves.png', dpi=150)
    plt.close()
    print("Zapisano: results/plots/calibration_curves.png")

def plot_roc_curves(X, y):
    """
    Krzywe ROC dla najlepszych modeli (train/test split).
    Pozwala wizualnie porównać zdolność modeli do separacji klas.
    """
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    roc_models = {
        'LogisticRegression (KNN imp)': Pipeline([
            ('imputer', KNNImputer(n_neighbors=5)),
            ('scaler', StandardScaler()),
            ('selector', SelectKBest(f_classif, k=6)),
            ('classifier', LogisticRegression(max_iter=1000, random_state=42))
        ]),
        'XGBoost (KNN imp)': Pipeline([
            ('imputer', KNNImputer(n_neighbors=5)),
            ('scaler', StandardScaler()),
            ('selector', SelectFromModel(
                RandomForestClassifier(n_estimators=100, random_state=42),
                max_features=6, threshold=-np.inf
            )),
            ('classifier', XGBClassifier(n_estimators=200, random_state=42,
                                         eval_metric='logloss', verbosity=0))
        ]),
        'RandomForest (Median imp)': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('selector', SelectFromModel(
                RandomForestClassifier(n_estimators=100, random_state=42),
                max_features=6, threshold=-np.inf
            )),
            ('classifier', RandomForestClassifier(n_estimators=200, random_state=42))
        ]),
    }

    fig, ax = plt.subplots(figsize=(8, 7))
    for name, pipe in roc_models.items():
        pipe.fit(X_train, y_train)
        RocCurveDisplay.from_estimator(pipe, X_test, y_test, ax=ax, name=name)

    ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
    ax.set_title('Krzywe ROC dla najlepszych konfiguracji', fontsize=13)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR + 'plots/roc_curves.png', dpi=150)
    plt.close()
    print("Zapisano: results/plots/roc_curves.png")


def plot_feature_importance(X, y):
    """
    Porównuje ranking cech wg SelectKBest (ANOVA F-score)
    i Random Forest Feature Importance.
    Pozwala zobaczyć, które cechy są wybierane przez każdą metodę.
    """
    imputer = SimpleImputer(strategy='median')
    X_imp = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    # SelectKBest ANOVA
    skb = SelectKBest(f_classif, k='all')
    skb.fit(X_imp, y)
    scores_anova = pd.Series(skb.scores_, index=X.columns).sort_values(ascending=True)

    # Random Forest Importance
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_imp, y)
    scores_rf = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].barh(scores_anova.index, scores_anova.values, color='steelblue')
    axes[0].set_title('SelectKBest – ANOVA F-score', fontsize=12)
    axes[0].set_xlabel('F-score')

    axes[1].barh(scores_rf.index, scores_rf.values, color='darkorange')
    axes[1].set_title('Random Forest – Feature Importance', fontsize=12)
    axes[1].set_xlabel('Importance')

    plt.suptitle('Porównanie metod selekcji cech', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR + 'plots/feature_importance.png', dpi=150)
    plt.close()
    print("Zapisano: results/plots/feature_importance.png")


# ====================== MAIN ======================
if __name__ == "__main__":
    # 1. Wczytanie danych
    df = load_data(DATA_PATH)
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']

    # 2. Eksperymenty
    results_df = run_experiments(X, y)

    # 3. Zapis wyników
    results_df_sorted = results_df.sort_values('F1', ascending=False)
    results_df_sorted.to_csv(RESULTS_DIR + 'results_comparison.csv', index=False)

    print("\n=== TOP 5 KONFIGURACJI (wg F1) ===")
    print(results_df_sorted[['Model', 'Imputer', 'Selector',
                             'F1', 'F1_std', 'ROC_AUC', 'ROC_AUC_std',
                             'Accuracy']].head(5).to_string(index=False))

    print("\n=== ANALIZA WPŁYWU IMPUTACJI (średnie po imputerach) ===")
    print(results_df.groupby('Imputer')[['F1', 'ROC_AUC', 'Accuracy']].mean().round(4))

    print("\n=== ANALIZA WPŁYWU SELEKCJI CECH (średnie po selektorach) ===")
    print(results_df.groupby('Selector')[['F1', 'ROC_AUC', 'Accuracy']].mean().round(4))

    print("\n=== ANALIZA STABILNOŚCI (std po modelach) ===")
    print(results_df.groupby('Model')[['F1_std', 'ROC_AUC_std']].mean().round(4))

    # 4. Wykresy
    print("\n=== GENERUJĘ WYKRESY ===")
    plot_comparison(results_df_sorted)
    plot_feature_importance(X, y)
    plot_calibration(X, y)
    plot_roc_curves(X, y)

    print("\nGotowe! Wszystkie wyniki w folderze 'results/'")