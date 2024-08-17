"""
Script Name: starter/ml/data.py
Author: Zidane
Date: 17-08-2024
"""
import numpy as np
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder

def process_data(
    X, categorical_features=[], label=None, training=True, encoder=None, lb=None
):
    """Process the data used in the machine learning pipeline."""
    if label is not None and label in X:
        y = X[label].copy()
        X = X.drop([label], axis=1)  # Remove label from features
    else:
        y = np.array([])

    X_categorical = X[categorical_features].copy()
    X_continuous = X.drop(columns=categorical_features)

    if training:
        # Initialize and fit the encoders during training
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        lb = LabelBinarizer()

        X_categorical = encoder.fit_transform(X_categorical)
        y = lb.fit_transform(y.values).ravel()
    else:
        if encoder is not None:
            missing_cols = set(encoder.feature_names_in_) - set(X_categorical.columns)
            for col in missing_cols:
                X_categorical[col] = ""

            X_categorical = X_categorical[encoder.feature_names_in_]
            X_categorical = encoder.transform(X_categorical)

            try:
                y = lb.transform(y.values).ravel()
            except AttributeError:
                pass
        else:
            raise ValueError("Encoder must be provided for inference.")

    X = np.concatenate([X_continuous.values, X_categorical], axis=1)
    return X, y, encoder, lb
