"""
Script Name: model.py
Purpose: This script provides a class-based implementation for training a machine learning model, evaluating its performance, and making predictions. The model used in this script is a RandomForestClassifier from scikit-learn.
Author: Zidane
Date: 08-08-2024
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score
import numpy as np
import pandas as pd
from .data import process_data

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
        return self.model.predict(X)

    def compute_slice_performance(self, X, y, feature, encoder, lb, output_file="slice_output.txt"):
        """
        Computes and writes model performance metrics for each unique value of a categorical feature.

        Inputs
        ------
        X : pd.DataFrame
            The data to compute metrics on.
        y : np.array
            The true labels.
        feature : str
            The categorical feature to slice by.
        encoder : OneHotEncoder
            The encoder used for categorical features.
        lb : LabelBinarizer
            The label binarizer used for the labels.
        output_file : str
            The name of the file to write the slice metrics to.
        """
        unique_values = X[feature].unique()
        with open(output_file, "w") as f:
            for value in unique_values:
                # Create a slice of data where the feature equals the current value
                slice_mask = X[feature] == value
                X_slice = X[slice_mask]
                y_slice = y[slice_mask]

                # Remove the label if still present in X_slice
                if 'salary' in X_slice.columns:
                    X_slice = X_slice.drop(columns=['salary'])

                # Process the data slice using the existing encoder and label binarizer
                X_processed, y_processed, _, _ = process_data(
                    X_slice, categorical_features=encoder.feature_names_in_, label=None, encoder=encoder, lb=lb, training=False
                )

                # Check if y_processed is empty
                if y_processed.size == 0:
                    continue  # Skip this slice if no labels are present

                # Make predictions
                preds = self.predict(X_processed)

                # Compute metrics
                precision, recall, fbeta = self.compute_metrics(y_processed, preds)

                # Write the metrics to the file
                f.write(f"Performance for {feature} = {value}:\n")
                f.write(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F-beta: {fbeta:.4f}\n\n")
