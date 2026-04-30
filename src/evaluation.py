from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def evaluate_model(model, X, y_true):
    """Zwraca Accuracy, F1-score i ROC-AUC."""

    y_pred = model.predict(X)

    if hasattr(model, "predict_proba"):
        y_scores = model.predict_proba(X)[:, 1]
    else:
        y_scores = model.decision_function(X)
        y_scores = (y_scores - y_scores.min()) / (y_scores.max() - y_scores.min())

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_scores)
    }
