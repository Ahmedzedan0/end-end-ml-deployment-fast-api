"""
Script Name: test_model.py
Purpose: This script contains unit tests for the ModelTrainer class, ensuring the correctness of the model training, evaluation, and inference functionalities. It uses pytest fixtures to efficiently manage test data and includes checks to verify that the model has been properly fitted.
Author: Zidane
Date: 14-08-2024
"""

import pytest
import numpy as np
import os
import sys
from sklearn.exceptions import NotFittedError

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '..'))

try:
    from ml.model import ModelTrainer  # Adjusted path
except ImportError:
    from model import ModelTrainer  # Alternative import


@pytest.fixture
def sample_data():
    """
    Pytest fixture to create sample data for testing.
    Returns
    -------
    X_train : np.array
        Sample training data.
    y_train : np.array
        Sample training labels.
    """
    X_train = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])
    y_train = np.array([1, 0, 1, 0])
    return X_train, y_train


@pytest.fixture
def fitted_model(sample_data):
    """
    Pytest fixture to return a fitted model.
    """
    X_train, y_train = sample_data
    trainer = ModelTrainer()
    trainer.train(X_train, y_train)
    return trainer


def test_train_model(fitted_model, sample_data):
    """
    Test whether the model is successfully trained and fitted.
    """
    model = fitted_model.model
    assert model is not None

    X_train, _ = sample_data
    try:
        model.predict(X_train)
    except NotFittedError:
        pytest.fail("Model is not fitted")


def test_compute_model_metrics():
    """
    Test the compute_model_metrics function to ensure correct calculation of precision, recall, and F1 score.
    """
    y = np.array([1, 0, 1, 0])
    preds = np.array([1, 0, 1, 1])
    trainer = ModelTrainer()
    precision, recall, fbeta = trainer.compute_metrics(y, preds)
    assert precision == 0.6666666666666666
    assert recall == 1.0
    assert fbeta == 0.8


def test_inference(fitted_model, sample_data):
    """
    Test the inference function to ensure it returns predictions of the correct length.
    """
    X_test, _ = sample_data
    preds = fitted_model.predict(X_test)
    assert preds is not None
    assert len(preds) == len(X_test)
