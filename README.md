# Diabetes Prediction

Projekt do klasyfikacji występowania cukrzycy na podstawie danych medycznych z pliku `data/diabetes.csv`.

W repozytorium zastosowano klasyczny pipeline uczenia maszynowego z preprocessingiem, selekcją cech, tuningiem hiperparametrów przez **Optuna** oraz oceną modeli w schemacie **nested cross-validation**.

## Zawartość projektu

- **Preprocessing danych**
  - zamiana biologicznie niemożliwych wartości `0` na `NaN` dla wybranych kolumn,
  - imputacja braków metodą `mean`, `median` albo `knn`,
  - skalowanie cech numerycznych,
  - selekcja `n_features` najlepszych cech z użyciem `mutual_info_classif`.
- **Modele klasyfikacyjne**
  - `LogisticRegression`
  - `SVM`
  - `RandomForestClassifier`
  - `GradientBoostingClassifier`
- **Optymalizacja**
  - strojenie hiperparametrów każdego modelu przez Optuna,
  - osobny dobór imputacji dla każdej iteracji outer CV.
- **Ewaluacja**
  - metryka główna: **ROC-AUC**,
  - walidacja krzyżowa typu nested CV.
- **Artefakty wynikowe**
  - zapis najlepszego modelu do `results/best_models/`,
  - zapis tabeli wyników do `results/final_results.csv`,
  - zapis wykresu porównawczego do `results/plots/model_comparison.png`,
  - zapis szczegółów foldów do `results/folds/`.

## Wymagania

- Python 3.10+,
- biblioteki z pliku `requirements.txt`.

## Instalacja

```bash
pip install -r requirements.txt
```

Warto wykonać instalację w wirtualnym środowisku:

```bash
python -m venv .venv
source .venv/bin/activate   # Linux / macOS
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

## Uruchomienie

Główny skrypt:

```bash
python main.py
```

Skrypt:

1. wczytuje dane z `data/diabetes.csv`,
2. uruchamia nested cross-validation dla każdego modelu,
3. zapisuje wyniki i najlepsze pipeline’y do katalogu `results/`.

## Konfiguracja

Najważniejsze parametry znajdują się w pliku `config.py`:

- `random_state = 42` — powtarzalność eksperymentów,
- `outer_cv = 3` — liczba foldów outer CV,
- `inner_cv = 2` — liczba foldów inner CV,
- `n_trials = 5` — liczba prób Optuna na każdą konfigurację,
- `n_features = 6` — liczba wybieranych cech,
- `impute_strategies = ["mean", "median", "knn"]` — testowane strategie imputacji.

## Wyniki

Po uruchomieniu projektu w katalogu `results/` pojawiają się pliki z wynikami. Dla obecnego uruchomienia najlepsze ROC-AUC uzyskał model **RandomForest**.

| Model | Średni ROC-AUC | Odchylenie standardowe | Najlepsza imputacja |
|---|---:|---:|---|
| RandomForest | 0.8375 | 0.0256 | knn |
| LogisticRegression | 0.8368 | 0.0197 | median |
| GradientBoosting | 0.8324 | 0.0174 | knn |
| SVM | 0.8211 | 0.0171 | median |

## Struktura katalogów

```text
.
├── config.py
├── main.py
├── requirements.txt
├── data/
│   └── diabetes.csv
├── src/
│   ├── evaluation.py
│   ├── models.py
│   ├── preprocessing.py
│   ├── training.py
│   └── utils.py
└── results/
    ├── best_models/
    ├── folds/
    ├── plots/
    └── final_results.csv
```

## Jak działa pipeline

### 1. Przygotowanie danych

W module `src/preprocessing.py` znajduje się własny transformer `ReplaceInvalidValues`, który zamienia wybrane wartości `0` na brak danych. Następnie dane przechodzą przez imputację, skalowanie i selekcję cech.

### 2. Strojenie modeli

W `src/models.py` zdefiniowano bazowe modele oraz ich przestrzenie hiperparametrów. Tuning odbywa się w `src/training.py` z użyciem Optuna.

### 3. Nested cross-validation

Każdy model jest oceniany w dwóch poziomach walidacji:

- **inner CV** — wybór najlepszych hiperparametrów,
- **outer CV** — uczciwa ocena końcowa.

### 4. Zapis wyników

Po zakończeniu treningu projekt zapisuje:

- najlepszy pipeline danego modelu w formacie `.joblib`,
- tabelę zbiorczą w CSV,
- wykres porównawczy modeli,
- szczegóły foldów w CSV i JSON.

## Pliki wynikowe

- `results/final_results.csv` — podsumowanie wszystkich modeli,
- `results/plots/model_comparison.png` — wykres porównania ROC-AUC,
- `results/best_models/*.joblib` — zapisane najlepsze modele,
- `results/folds/*` — szczegóły poszczególnych foldów.

## Uwaga

Projekt jest nastawiony na eksperymenty porównawcze. Najprościej uruchamiać go przez `main.py`, a konfigurację zmieniać w `config.py`.
