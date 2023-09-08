# Validation functions


def validate_features_to_hide(X, to_hide):
    max_value = X.shape[1]

    if all(isinstance(i, int) for i in to_hide) and max(to_hide) < max_value:
        return True
    return False
