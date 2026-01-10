from lightgbm import LGBMClassifier


def build_lightgbm_model(params: dict) -> LGBMClassifier:
    return LGBMClassifier(
        n_estimators=300 if 'n_estimators' not in params else params['n_estimators'],
        learning_rate=0.05 if 'learning_rate' not in params else params['learning_rate'],
        max_depth=-1 if 'max_depth' not in params else params['max_depth'],
        num_leaves=31 if 'num_leaves' not in params else params['num_leaves'],
        subsample=0.8 if 'subsample' not in params else params['subsample'],
        colsample_bytree=0.8 if 'colsample_bytree' not in params else params['colsample_bytree'],
        class_weight="balanced" if 'class_weight' not in params else params ['class_weight'],
        random_state=42 if 'random_state' not in params else params['random_state'],
        n_jobs=-1 if 'n_jobs' not in params else params['n_jobs'],
    )
