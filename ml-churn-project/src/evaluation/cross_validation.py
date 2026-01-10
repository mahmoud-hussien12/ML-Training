from sklearn.model_selection import StratifiedKFold, cross_val_score


def evaluate_with_cv(pipeline, X, y):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(
        pipeline,
        X,
        y,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
    )

    return scores.mean(), scores.std()
