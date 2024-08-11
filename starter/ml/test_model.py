"""
Script Name: test_model.py
Author: Zidane
Date: 08-08-2024
"""
import pytest
import numpy as np
from model import train_model, compute_model_metrics, inference

def test_train_model():
    X_train = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])
    y_train = np.array([1, 0, 1, 0])
    model = train_model(X_train, y_train)
    assert model is not None

def test_compute_model_metrics():
    y = np.array([1, 0, 1, 0])
    preds = np.array([1, 0, 1, 1])
    precision, recall, fbeta = compute_model_metrics(y, preds)
    assert precision == 0.6666666666666666
    assert recall == 1.0
    assert fbeta == 0.8

def test_inference():
    X_test = np.array([[1, 0], [0, 1]])
    y_train = np.array([1, 0, 1, 0])
    model = train_model(X_test, y_train)
    preds = inference(model, X_test)
    assert preds is not None
    assert len(preds) == len(X_test)
