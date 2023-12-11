"""
Module defining class to apply the sarima model to multiple time series
in parallel.
"""
import numpy as np
from pmdarima import arima

class MultivariateSARIMA():
    """
    Class defining a multivariate forcaster that implements SARIMA models
    on each series on the data set.
    """
    def __init__(self,
        start_p:int=0,
        start_q:int=0,
        max_p:int=12,
        max_q:int=12,
        d:int=1,
        seasonal:bool=False,
        test:str='adf'
    ):
        """
        Class instantiation method.

        Args
        ----------
            start_p:int=0
                Minimum autorregresive order for the values
            start_q:int=0
                Minimum autorregresive order for the errors
            max_p:int=12
                Maximum autorregresive order for the values
            max_q:int=12
                Maximum autorregresive order for the errors
            d:int=1
                Differentiation order
            seasonal:bool=False
                Either there's a seasonal component
            test:str='adf'
                Type of test to be conducted
        """
        self.start_p=start_p
        self.start_q=start_q
        self.max_p=max_p
        self.max_q=max_q
        self.d=d
        self.seasonal=seasonal
        self.test=test

    def fit(self, data:np.ndarray):
        """
        Method for the automatic instantiation and automaticautorregresive order
        selection for the models for each series in the data using the auto_arima
        function.
        
        Args:
        ----------
            data:np.ndarray
                2D array containing the time series, with each searies as a column
        """
        cols = data.shape[1]
        self.models = {}
        self.fitted_values=[]

        for i in range(cols):
            self.models["sarima"+str(i)] = arima.auto_arima(
                y=data[:,i],
                start_p=self.start_p,
                start_q=self.start_q,
                max_p=self.max_p,
                max_q=self.max_q,
                d=self.d,
                seasonal=self.seasonal,
                test=self.test
            )
            self.fitted_values.append(self.models["sarima"+str(i)].fittedvalues())

        self.fitted_values = np.array(self.fitted_values).transpose()

    def predict(self, h:int=12):
        """
        Method to forecast an specific amount of timesteps for all time series

        Args:
        ----------
            h:int
                Number of timesteps to forecast

        Returns:
        ----------
            np.ndarray
                2D array containing all forecasts.
        """
        forecasts = [model.predict(h) for model in self.models.values()]
        return np.array(forecasts).transpose()