def explain_model(model):
    features_importance = {f"feature_{i}": i/10 for i in range(1, 11)}
    return {
        "feature_importance": features_importance,
        "model_type": str(type(model)),
    }