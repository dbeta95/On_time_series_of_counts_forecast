"""
Module with general porpose function for the simulation study.
"""

import numpy as np
import matplotlib.pyplot as plt

def plot_simulations(
    sims:np.array,
    cols:int,
    x_size:int=8,
    y_size:int=4
) -> None:
    """
    Function to plot the simulated time series in an array.

    Args:
    ---------
        sims:np.array
            Array containing all simulated series, each as a column
        cols:int
            Number of columns to use in the time series plotting
        x_size:int=8
            Horizontal size for each plot
        y_size:int=4
            Vertial size for each plot
    """
    rows = int(np.ceil(sims.shape[1]/cols))

    fig, axs = plt.subplots(rows, cols, figsize=(cols*x_size, rows*y_size))
    for row in range(rows):
        for col in range(cols):
            try:
                axs[row, col].plot(sims[:, cols*row+col])
                axs[row, col].set_title(f'Simulation {cols*row+col+1}')
            except:
                pass
    plt.show()

def plot_fit(true_values:np.ndarray, fitted_values:np.ndarray, predicted:bool=False, cols:int=4):
    """
    Function to plot the real and fitted values for all timeseries in the dataset

    Args:
    ----------
        true_values:np.ndarray
            True observed values for the time series
        fitted_values:np.ndarray
            Values fitted by the model
        predicted:bool=False
            Either it the values are fitted o predictions
        cols:int
            Number of columns to use in the time series plotting
    """   
    rows = int(np.ceil(true_values.shape[1]/cols))
    label = 'Fitted'
    if predicted:
        label = 'Predicted'

    fig, axs = plt.subplots(rows, cols, figsize=(cols*8, rows*4))
    for row in range(rows):
        for col in range(cols):
            try:
                axs[row, col].plot(true_values[:, cols*row+col])
                axs[row, col].plot(fitted_values[:, cols*row+col])
                axs[row, col].set_title(f'Series {cols*row+col+1}')
                axs.legend(["Actual", label])
            except:
                pass
    plt.show()

def process_simulated_series(series:np.ndarray, seed:int=None):
    """
    Funciton to process a simulated data to rescale securing a certain
    percentage of zeros ocurrence and convert into count values.

    Args:
    ----------
        series:np.ndarray
            Time series to be processed
        seed:int=None
            Random seed for the percentage of zeros draw

    Returns:
    ---------
        np.ndarray
            Processed time series
    """
    if seed:
        np.random.seed(seed)

    quatile = np.random.random()*0.05+0.05
    min_val = np.quantile(series, quatile)
    series = series - min_val
    series = np.where(series < 0, 0, series)
    series = np.round(series, decimals=0)
    return series

def process_simulated_data(data:np.ndarray, seed:int=None):
    """
    Funciton to process the simulated data to rescale securing a certain
    percentage of zeros ocurrence and convert into count values.

    Args:
    ----------
        series:np.ndarray
            Time series dataset to be processed
        seed:int=None
            Random seed for the percentage of zeros draw

    Returns:
    ---------
        np.ndarray
            Processed time series dataset
    """
    if seed:
        np.random.seed(seed)
    return np.apply_along_axis(process_simulated_series, 0 ,data)
