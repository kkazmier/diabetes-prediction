from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def evaluate_model(model, X, y_true) -> float:
    """Zwraca ROC-AUC."""
    if hasattr(model, "predict_proba"):
        y_pred = model.predict_proba(X)[:, 1]
    else:
        y_pred = model.decision_function(X)
        y_pred = (y_pred - y_pred.min()) / (y_pred.max() - y_pred.min())

    return roc_auc_score(y_true, y_pred)
