def get_feature_columns(cfg):
    """
    Returns:
    - numerical_features: list[str]
    - categorical_features: list[str]
    """
    numerical_features = cfg['data']['features']['numerical_features']
    categorical_features = cfg['data']['features']['categorical_features']
    return numerical_features, categorical_features
