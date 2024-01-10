"""
Module with self defined function to be used as forecast
metrics
"""

import numpy as np
from sklearn.metrics import mean_absolute_error

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
    error = np.abs(y_pred-y_true)
    norm = np.maximum(np.abs(y_true) + np.abs(y_pred), epsilon)
    return 2*np.mean(error/norm, axis=0)*100

def mean_absolute_scaled_error(y_true,y_pred,in_sample,m:int):
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
        m:int
            the time interval betwwen seccesive observations for each frequecy
    """
    if not all([
        isinstance(y_true, np.ndarray), 
        isinstance(y_pred, np.ndarray),
        isinstance(in_sample, np.ndarray)
    ]): 
        y_true, y_pred, in_sample = np.array(y_true), np.array(y_pred), np.array(in_sample)

    error = np.abs(y_pred-y_true)
    train_mae = mean_absolute_error(in_sample[m:], in_sample[:-m], multioutput="raw_values")

    return np.mean(error/train_mae, axis=0)

def get_model_metrics(    
    test_data:np.ndarray,
    predicted_values:np.ndarray,
    train_data:np.ndarray,
    m:int,
    model_name:str=None,
    case:str=None
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
        m:int
            the time interval betwwen seccesive observations for each frequecy
        model_name:str=None
            Name of the model which the predictions where generated with.

    Returns:
    ----------
        metrics:dict
            Dictionary containing the sMAPE and MASE for all time series
        statistics:dict
            Dictionary containing summay statistics for the sMAPE and MASE
    """
    smape = symmetric_mean_absolute_percentage_error(
        test_data, predicted_values
    )
    mase = mean_absolute_scaled_error(
        test_data, predicted_values, train_data, m
    )

    metrics = {
        'sMAPE':smape,
        'MASE':mase
    }

    statistics = {
        'sMAPE':{
            'mean':np.mean(smape),
            'sd':np.std(smape),
            'min':np.min(smape),
            'max':np.max(smape)
        },
        'MASE':{
            'mean':np.mean(mase),
            'sd':np.std(mase),
            'min':np.min(mase),
            'max':np.max(mase)
        }
    }

    if model_name is not None and case is None:
        raise ValueError("If a model name is defined the case has to be defined as well")
        
    if case is not None and model_name is None:
        raise ValueError("If a case is defined the model name has to be defined as well")
        
    if model_name is not None:

        metrics = {
            (f"{model_name}-{case}", metric): values for metric, values in metrics.items()
        }

        statistics = {
            (model_name, metric):{
                (case, statistic):value for statistic, value in inner_dict.items()
            } for metric, inner_dict in statistics.items()
        }

    return metrics, statistics