"""
Module with self defined function to be used as forecast
metrics
"""

import numpy as np

def symmetric_mean_absolute_percentage_error(y_true,y_pred):
    """
    Symmetric mean absolute percentage error (SMAPE) for regression.

    Note that the result is a percentage within a range from [0,200]

    Args:
    ----------
        y_true: Matrix_like|Array_like
            Ground truth target values
        y_pred: Matrix_like|Array_like
            Estimated target values
    """
    if not all([isinstance(y_true, np.ndarray), isinstance(y_pred, np.ndarray)]): 
        y_true, y_pred = np.array(y_true), np.array(y_pred)

    epsilon = np.finfo(np.float64).eps
    smape = np.mean(
        np.abs(y_pred-y_true)/
        (np.maximum(np.abs(y_true) + np.abs(y_pred) + epsilon, 0.5 + epsilon)/2)
    )*100

    return smape


def mean_absolute_scaled_error(y_true,y_pred,in_sample):
    """
    Mean absolute scaled error as proposed in  https://robjhyndman.com/papers/mase.pdf

    Args:
    ----------
        y_true: Matrix_like|Array_like
            Ground truth target values
        y_pred: Matrix_like|Array_like
            Estimated target values
        in_sample: Matrix_like|Array_like
            Sample data for training
    """
    if not all([
        isinstance(y_true, np.ndarray), 
        isinstance(y_pred, np.ndarray),
        isinstance(in_sample, np.ndarray)
    ]): 
        y_true, y_pred, in_sample = np.array(y_true), np.array(y_pred), np.array(in_sample)

    e = np.abs(y_pred-y_true)
    n = np.prod(in_sample.shape)
    m = np.prod(in_sample.shape[0])
    norm = np.sum(np.abs(in_sample[1:]-in_sample[:-1]))/(n-m)

    mase = np.mean(e/norm)

    return mase

def get_model_metrics(    
    test_data:np.ndarray,
    predicted_values:np.ndarray,
    train_data:np.ndarray,
):
    """
    Function to compute the metrics to evaluate the models performance
    by computing the SMAPE and the MASE for the forecasted values.

    Args:
    ----------
        test_data:np.ndarray
            2D array containing the test data
        predicted_values:np.ndarray
            2D array containing the fitted values
        train_data:np.ndarray
            2D array containing the timeseries used to fit

    Returns:
    ----------
        smape:float
            SMAPE of the scaled predicted values
        mase:float
            MASE of the scaled predicted values
    """
    smape = symmetric_mean_absolute_percentage_error(
        test_data, predicted_values
    )
    mase = mean_absolute_scaled_error(
        test_data, predicted_values, train_data
    )

    print(f"SMAPE: {smape}")
    print(f"MASE: {mase}")

    return smape, mase