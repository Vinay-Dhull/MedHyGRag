from sklearn.metrics import accuracy_score, f1_score

LABELS = ("yes", "no", "maybe")


def classification_scores(y_true: list[str], y_pred: list[str]) -> dict[str, float]:
    return {
        "n": len(y_true),
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "macro_f1": round(
            f1_score(y_true, y_pred, labels=list(LABELS), average="macro", zero_division=0), 4
        ),
    }
