"""
Script Name: model.py
Purpose: This script provides a class-based implementation for training a machine learning model,
evaluating its performance, and making predictions.
The model used in this script is a RandomForestClassifier from scikit-learn.
Author: Zidane
Date: 18-08-2024
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.utils.validation import check_is_fitted, NotFittedError
import numpy as np
import pandas as pd
from .data import process_data  # Adjusted import for process_data


class ModelTrainer:
    """
    A class to encapsulate the machine learning model training, evaluation, and inference process.
    """

    def __init__(self):
        """
        Initializes the ModelTrainer with a RandomForestClassifier.
        """
        self.model = RandomForestClassifier()

    def train(self, X_train, y_train):
        """
        Trains the machine learning model.

        Inputs
        ------
        X_train : np.array
            Training data.
        y_train : np.array
            Labels.
        """
        self.model.fit(X_train, y_train)

    def compute_metrics(self, y_true, y_pred):
        """
        Validates the trained model using precision, recall, and F1 score.

        Inputs
        ------
        y_true : np.array
            Known labels, binarized.
        y_pred : np.array
            Predicted labels, binarized.

        Returns
        -------
        precision : float
        recall : float
        fbeta : float
        """
        fbeta = fbeta_score(y_true, y_pred, beta=1, zero_division=1)
        precision = precision_score(y_true, y_pred, zero_division=1)
        recall = recall_score(y_true, y_pred, zero_division=1)
        return precision, recall, fbeta

    def predict(self, X):
        """
        Run model inferences and return the predictions.

        Inputs
        ------
        X : np.array
            Data used for prediction.

        Returns
        -------
        preds : np.array
            Predictions from the model.
        """
        try:
            check_is_fitted(self.model)
        except NotFittedError:
            raise NotFittedError(
                "The RandomForestClassifier instance is not fitted yet. "
                "Call 'train' with appropriate arguments before using this method."
            )

        return self.model.predict(X)
