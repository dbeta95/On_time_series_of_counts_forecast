"""
Module containing the function to create the simulation scenarios
for the comparison between models
"""
import numpy as np

def simulate_ar_poisson_series(
    lenght:int,
    max_dependency:int,
    seed:int = None
):
    """
    Function to simulate an Autorregresive Poisson time series.

    Args:
    ----------
        lenght:int
            Number of time steps in the time series
        max_dependency:int
            Maximum lag for the dependency on both the mean and previous values
        seed:int
            Random seed for the simulation

    Returns:
    ----------
        np.array
            Array containing the timeseries
    """
    if seed:
        np.random.seed(seed)

    p = np.random.randint(1,max_dependency)
    q = np.random.randint(1,max_dependency)

    coefs_nus = np.random.uniform(-1,1,p)
    coefs_values = np.random.uniform(-1,1,q)
    norm = np.abs(coefs_nus).sum() + np.abs(coefs_values).sum()
    coefs_nus = coefs_nus/norm
    coefs_values = coefs_values/norm
    intercept = np.random.uniform(0,5)
    init_mean = np.random.randint(0,10)
    sigma2 = np.random.uniform(0,1)
    means = np.random.poisson(init_mean, p)
    nus = np.log(means)
    values = np.random.poisson(init_mean, q)

    for _ in range(lenght):
        arg_1 = np.sum(nus[-coefs_nus.shape[0]:]*coefs_nus)
        arg_2 = np.sum(np.log(values[-coefs_values.shape[0]:]+1)*coefs_values)
        nu = arg_1 + arg_2 + intercept + np.random.normal(loc=0, scale=sigma2)
        mean = np.exp(nu)
        value = np.random.poisson(mean)
        nus = np.append(nus, nu)
        values = np.append(values, value)

    return values[q:]


def simulate_independent_series(
    n:int,
    lenght:int,
    max_dependency:int,
    seed:int = None
):
    
    """
    Function to simulate an Autorregresive Poisson time series.

    Args:
    ----------
        n:int   
            Number of series to be simulated
        lenght:int
            Number of time steps in the time series
        max_dependency:int
            Maximum lag for the dependency on both the mean and previous values
        seed:int
            Random seed for the simulation

    Returns:
    ----------
        np.array
            Array containing the timeseries
    """
    if seed:
        np.random.seed(seed)

    simulations = np.zeros((lenght, n))
    n_sim = 0

    while n_sim < n:
        try:
            sim = simulate_ar_poisson_series(lenght, max_dependency)
            simulations[:, n_sim] = sim
            n_sim += 1
        except:
            pass

    return simulations