"""
Module containing the classes related with the implementation of the auto regressive
Poisson model for forecasting time series of counts, also named IGARCH.
"""

import os
import sys
import warnings

import numpy as np
from typing import Optional, Any
from statsmodels.tools.validation import array_like
from scipy.optimize import minimize, LinearConstraint
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

home_path = os.path.dirname(os.path.dirname(__file__))
sys.path.append(home_path)

from Optimization.optimizers import Adam

class PoissonAutoregression():

    def __init__(self, p:int, q:int, link:Optional[str]="log-lineal") -> None:
        """
        Class instantiation method

        Args
        ----------
            p:int
                Order of the autorregressive polinomy for the rate.
            q:int
                Order of the autorregressive polinomy for the series values.
            link:Optional[str]
                Link function for the mean and the systematic component. Can be
                "lineal" or "log-lineal"
        """
        if link.lower() not in ("lineal", "log-lineal"):
            raise ValueError(
                'The model type %r is not a valid option. Please select either the "lineal" or "log-lineal" model type'
            )
        self.type = link
        self.p = p
        self.q = q
        self.n_par = self.p+self.q+1

    def __initialize_values__(self,y:Any,theta:np.ndarray, epsilon:float=1e-8):
        """
        Abstract method to initialice the values for the rates and covariables.
        The method depends on the type of model used and varies in the variable 
        initialized for the systematic component of the model. For the lineal model
        lambda values corresponding to the Poisson proccess rate will be used, whereas 
        for the log-lineal model nu values corresponding to the logarithm of the 
        rate will be used.

        Args
        ----------
            y: Any
                One dimentional array-like object containing the series observations
            theta: nd.array
                One dimentional array containing th emodel coefficients
        """

        if self.type == 'lineal':
            # Initializing the variable objects
            self.y = array_like(y, 'y')
            self.n = self.y.shape[0]
            self.X = np.zeros(shape=(self.n,self.n_par))
            self.lmbd = np.zeros(shape=(self.n,))
            self.grad_lmbd = np.zeros(shape=(self.n,self.n_par))

            # Setting the covariables for the model
            self.k = max(self.p,self.q)

            self.lmbd[:self.k] = self.y[:self.k]
            for t in range(self.k,self.n):

                lmbds = np.flip(self.lmbd[t-self.p:t])
                y = np.flip(self.y[t-self.q:t])            
                grad_lmbd_tp_t = np.flip(self.grad_lmbd[t-self.p:t,:],axis=0)

                x = np.append(np.ones(1), lmbds)
                x = np.append(x, y)
                grad_x = np.append(np.zeros((1,self.n_par)), grad_lmbd_tp_t, axis=0)
                grad_x = np.append(grad_x,np.zeros((self.q,self.n_par)), axis=0)


                lmbd = x.T @ theta
                grad_lmbd = x + grad_x.T @ theta

                self.X[t,:] = x
                self.lmbd[t] = lmbd
                self.grad_lmbd[t,:] = grad_lmbd

        if self.type == 'log-lineal':
            # Initializing the variable objects
            self.y = array_like(y, 'y')
            self.n = self.y.shape[0]
            self.X = np.zeros(shape=(self.n, self.n_par))
            self.nu = np.zeros(shape=(self.n,))
            self.grad_nu = np.zeros(shape=(self.n, self.n_par))
    
            # Setting the corabiables for the model
            self.k = max(self.p, self.q)
            self.nu[:self.k] = np.log(self.y[:self.k]+epsilon)
            for t in range(self.k, self.n):

                nus = np.flip(self.nu[t-self.p:t])
                y = np.log(np.flip(self.y[t-self.q:t]) + 1.)
                grad_nu_tp_t = np.flip(self.grad_nu[t-self.p:t,:], axis=0)

                x = np.append(np.ones(1), nus)
                x = np.append(x, y)
                grad_x = np.append(np.zeros((1,self.n_par)), grad_nu_tp_t, axis=0)
                grad_x = np.append(grad_x,np.zeros((self.q,self.n_par)), axis=0)

                nu = x.T @ theta
                grad_nu = x + grad_x.T @ theta

                self.X[t,:] = x
                self.nu[t] = nu
                self.grad_nu[t,:] = grad_nu

    def __update_values__(self,theta:np.ndarray):
        """
        Abstract method to update the values for the rates and covariables after
        each change in the parameters vector

        Args
        ----------
           theta: nd.array
                One dimentional array containingthemodel coefficients
        """
        if self.type == 'lineal':
            for t in range(self.k,self.n):

                lmbds = np.flip(self.lmbd[t-self.p:t])
                y = np.flip(self.y[t-self.p:t])            
                grad_lmbd_tp_t = np.flip(self.grad_lmbd[t-self.p:t,:],axis=0)

                x = np.append(np.ones(1), lmbds)
                x = np.append(x, y)
                grad_x = np.append(np.zeros((1,self.n_par)), grad_lmbd_tp_t, axis=0)
                grad_x = np.append(grad_x,np.zeros((self.q,self.n_par)), axis=0)


                lmbd = x.T @ theta
                grad_lmbd = x + grad_x.T @ theta

                self.X[t,:] = x
                self.lmbd[t] = lmbd
                self.grad_lmbd[t,:] = grad_lmbd

        if self.type == 'log-lineal':
            for t in range(self.k, self.n):

                nus = np.flip(self.nu[t-self.p:t])
                y = np.log(np.flip(self.y[t-self.q:t]) + 1.)
                grad_nu_tp_t = np.flip(self.grad_nu[t-self.p:t,:], axis=0)

                x = np.append(np.ones(1), nus)
                x = np.append(x, y)
                grad_x = np.append(np.zeros((1,self.n_par)), grad_nu_tp_t, axis=0)
                grad_x = np.append(grad_x,np.zeros((self.q,self.n_par)), axis=0)

                nu = x.T @ theta
                grad_nu = x + grad_x.T @ theta

                self.X[t,:] = x
                self.nu[t] = nu
                self.grad_nu[t,:] = grad_nu

    def __loss_function__(self, epsilon:float=1e-8):
        """
        Abstract method to calculate the loss function given by the conditional
        negative log likelihood function

        Args
        ---------
            epsilon:float=1e-8
                Stabilization factor to avoid math errors
        """
        if self.type == 'lineal':
            loss = (
                self.lmbd[self.k:] - self.y[self.k:]*np.log(self.lmbd[self.k:] + epsilon)
            ).sum()
            return loss
        
        if self.type == 'log-lineal':
            y_hat = 1/(np.exp(-self.nu[self.k:])+epsilon)
            loss = (                
                y_hat - self.y[self.k:]*self.nu[self.k:]
            ).sum()
            return loss
        
    def __grad__(self, epsilon:float=1e-8):
        """
        Abstract method to calculate the gradient vector of the loss function
        over the parameters

        Args
        ---------
            epsilon:float=1e-8
                Stabilization factor to avoid math errors
        """
        if self.type == 'lineal':
            grad = self.grad_lmbd[self.k:,:].T @ (1-self.y[self.k:]/(self.lmbd[self.k:]+epsilon))
            return grad

        if self.type == 'log-lineal':
            y_hat = 1/(np.exp(-self.nu[self.k:])+epsilon)
            grad = self.grad_nu[self.k:,:].T @ (y_hat - self.y[self.k:])
            return grad
        
    
    def fit(self,
        y:Any,
        method:Optional[str]='Adam',
        tol:Optional[float]=1e-6,
        num_iter:Optional[int]=10000
    ):
        """
        Method to estimate the parameters values by solving the minimize problem.

        Args:
        ----------
        y:Any
            One dimentional array-like object containing the series observations
        method:Optional[str]='Adam'
            Method used to find the parameters optimal values. For the log-lineal model
            only the Adam method is available.

            The available methods for the lineal model are "Adam" gradient descent and 
            "trust-constr" or "SLSQP" for the implementation in scipy 
            (see https://docs.scipy.org/doc/scipy/tutorial/optimize.html#constrained-minimization-of-multivariate-scalar-functions-minimize)
            for the Trust-Region Constrained Algorithm and the Sequential Least Squares Programming Algorithm
        
            Note that the "Adam" gradient descent is an unconstrained method and may fall to
            find values that fulfill the condition to garantee stationarity for the series, in which case
            the warning will shown.

        tol:Optional[float]=1e-6
            Bound of change to stop the algorithm. Only required for the Adam method
        num_iter:Optional[int]=10000
            Maximum number of iterations. Only required for the Adam method
                 
        """       

        if self.type == 'lineal':

            if method.lower() not in ['adam', 'trust-constr', 'slsqp']:
                raise ValueError(
                    'The selected method is not available. Available methods for the lineal model are "Adam", "trust-constr" and "SLSQP"'
                )

            if method == 'Adam':
                theta = np.concatenate([np.ones(1), np.zeros(self.n_par-1)])
                self.__initialize_values__(y,theta)
                self.hist = np.full(num_iter,np.nan)
                gradient_descent = Adam()

                for iter in range(num_iter):
                    grad_theta = self.__grad__()
                    theta = gradient_descent.update(theta=theta, grad_theta=grad_theta)
                    
                    self.__update_values__(theta)
                    self.hist[iter] = self.__loss_function__()

                    if np.abs(self.hist[iter]-self.hist[iter-1]) < tol:
                        break

                self.coefs = theta
                self.hist = self.hist[~np.isnan(self.hist)]
                self.fitted = self.lmbd

                if self.coefs[1:].sum() < 0 or self.coefs[1:].sum() > 1:
                    warnings.warn(
                        "The coefficients found doesn't full fill the condition for stationarity \n\
    and using a constrained method is adviced."
                    )

            if method == 'trust-constr':
                def f(theta):
                    self.__initialize_values__(y,theta)
                    return self.__loss_function__()
                
                def grad(theta):
                    self.__initialize_values__(y,theta)
                    return self.__grad__()
                
                linear_constraint=LinearConstraint(
                    np.concatenate([np.zeros(1), np.ones(self.n_par-1)]),
                    [0.],
                    [1.]
                )

                theta0 = np.concatenate([np.ones(1), np.zeros(self.n_par-1)])

                res = minimize(f,x0=theta0, 
                            method='trust-constr',
                            constraints=linear_constraint,
                            jac=grad)
                
                if res.success == False:
                    warnings.warn(
                        'Method "trust-constr" failed to converge to a solution with the message: \n %r'
                        % (res.message)
                    )
                theta = res.x

                self.__initialize_values__(y,theta)
                self.coefs = theta
                self.fitted = self.lmbd

            if method == 'SLSQP':
                
                def f(theta):
                    self.__initialize_values__(y,theta)
                    return self.__loss_function__()

                cons = (
                    {
                        'type':'ineq',
                        'fun':lambda theta: theta[1:].sum()
                    },
                    {
                        'type':'ineq',
                        'fun':lambda theta: 1 - theta[1:].sum()
                    }
                )

                theta0 = np.concatenate([np.ones(1), np.zeros(self.n_par-1)])

                res = minimize(f,x0=theta0, constraints=cons)
                if res.success == False:
                    warnings.warn(
                        'Method "SLSQP" failed to converge to a solution with the message: \n %r'
                        % (res.message)
                    )
                theta = res.x

                self.__initialize_values__(y,theta)
                self.coefs = theta
                self.fitted = self.lmbd

        if self.type == 'log-lineal':
            if method.lower() != 'adam':
                raise ValueError(
                    'The selected method is not available. Only "Adam" method is available for the log-lineal model'
                )
            
            theta = np.concatenate([np.ones(1), np.zeros(self.n_par-1)])
            self.__initialize_values__(y,theta)
            self.hist = np.full(num_iter,np.nan)
            gradient_descent = Adam()

            for iter in range(num_iter):
                grad_theta = self.__grad__()
                theta = gradient_descent.update(theta=theta, grad_theta=grad_theta)
                
                self.__update_values__(theta)
                self.hist[iter] = self.__loss_function__()

                if np.abs(self.hist[iter]-self.hist[iter-1]) < tol:
                    break

            self.coefs = theta
            self.hist = self.hist[~np.isnan(self.hist)]
            self.fitted = np.exp(self.nu)

    def predict(self, h:int):
        """
        Method to forecast h values after fitting the model's equation
        
        Args:
        ----------
            h:int
                Number of values in the future to forecast

        Results:
        ----------
            np.ndarray
                One dimentional array containing the forecasted values
        """
        if h < 1:
            raise ValueError(
                "The number of periods to forecast must be a value greater than zero."
            )
        
        if self.type == 'lineal':
            # Initializing the forecasts object
            self.y_pred = np.zeros(h)

            for t in range(h):

                lmbds_index = t - (np.arange(self.p)+1)
                lmbds = np.where(lmbds_index < 0, self.fitted[lmbds_index], self.y_pred[lmbds_index])
                z_index = t - (np.arange(self.q)+1)
                z = np.where(z_index < 0, self.y[z_index], self.y_pred[z_index])

                x = np.append(np.ones(1), lmbds)
                x = np.append(x, z)

                lmbd = x.T @ self.coefs
                self.y_pred[t] = lmbd
            
        if self.type == 'log-lineal':
            # Initializing the forecasts object
            self.y_pred = np.zeros(h)
            nus_v = np.zeros(h)

            for t in range(h):

                nus_index = t - (np.arange(self.p)+1)
                nus = np.where(nus_index < 0, self.nu[nus_index], nus_v[nus_index])
                z_index = t - (np.arange(self.q)+1)
                z = np.where(z_index < 0, np.log(self.y[z_index] + 1.), nus_v[z_index])

                x = np.append(np.ones(1), nus)
                x = np.append(x, z)

                nu = x.T @ self.coefs
                nus_v[t] = nu
            
            self.y_pred = np.exp(nus_v)

        return self.y_pred

def ts_cv_score(
    model:Any,
    series:Any, 
    loss_function:Optional[Any]= mean_squared_error, 
    folds:Optional[int]=5,
    **kwargs
):
    """
    Function to get de Cross-validated implamentation of an score for a time series
    given a model.

    Args:
    ----------
        model:Any
            Initialized model to be evaluated. Note that the model's object is required
            to have a "fit" method to train the model and a "predict" method to get a
            forecast with test data.
        series:Any
            One dimentional array-like object containing the series observations
        loss_function:Optional[function]= mean_squared_error
            Error metric to be minimized
        folds:Optional[int]=5
            number of train-test splits to be used
        **kwargs:
            Keyword arguments to be passed to the fit method of the model
    """
    errors = np.array([])
    series = array_like(series, "series")

    tscv = TimeSeriesSplit(n_splits=folds)
    
    for train, test in tscv.split(series):

        model.fit(series[train], **kwargs)
        h = len(test)

        predictions = model.predict(h)
        actual = series[test]

        error = loss_function(predictions, actual)
        errors = np.append(errors, error)

    return errors.mean()


class AutoINGARCH():

    def __init__(self,
        series:Any,
        max_p:Optional[int]=None,
        max_q:Optional[int]=None,
        cv_folds:Optional[int]=5,
        link:Optional[str]="log-lineal",
        method:Optional[str]='Adam',
        tol:Optional[float]=1e-6,
        num_iter:Optional[int]=10000,
        verbose:Optional[bool]=False
    ):
        """
        Class instantiation method

        Args:
        ----------
            series:Any
                One dimentional array-like object containing the series observations
            max_p:Optional[int]=None
                Maximum order of the lag over the rate of the process
            max_q:Optional[int]=None
                Maximum order of the lag over the previous observations
            cv_folds:Optional[int]=5
                Number of train-test splits to be used in the cross-validation
            link:Optional[str]
                Link function for the mean and the systematic component. Can be
                "lineal" or "log-lineal"
            method:Optional[str]='Adam'
                Method used to find the parameters optimal values. For the log-lineal model
                only the Adam method is available.

                The available methods for the lineal model are "Adam" gradient descent and 
                "trust-constr" or "SLSQP" for the implementation in scipy 
                (see https://docs.scipy.org/doc/scipy/tutorial/optimize.html#constrained-minimization-of-multivariate-scalar-functions-minimize)
                for the Trust-Region Constrained Algorithm and the Sequential Least Squares Programming Algorithm
            
                Note that the "Adam" gradient descent is an unconstrained method and may fall to
                find values that fulfill the condition to garantee stationarity for the series, in which case
                the warning will shown.

            tol:Optional[float]=1e-6
                Bound of change to stop the algorithm. Only required for the Adam method
            num_iter:Optional[int]=10000
                Maximum number of iterations. Only required for the Adam method
            verbose:Optional[bool]=False
                Wether or not print information for each iteration
            
        """
        self.series = array_like(series, 'series')
        self.max_p = max_p
        self.max_q = max_q
        self.folds = cv_folds

        if self.max_p is None:
            self.max_p = np.floor(self.series.shape[0]/(self.folds+1)).astype(int)

        if self.max_q is None:
            self.max_q = np.floor(self.series.shape[0]/(self.folds+1)).astype(int)

        p_orders = range(1,self.max_p+1)
        q_orders = range(1,self.max_q+1)

        self.grid = [(p,q) for p in p_orders for q in q_orders]

        if link.lower() not in ("lineal", "log-lineal"):
            raise ValueError(
                'The model type %r is not a valid option. Please select either the "lineal" or "log-lineal" model type'
            )
        
        self.link = link

        if self.link == 'lineal' and method.lower() not in ['adam', 'trust-constr', 'slsqp']:
            raise ValueError(
                'The selected method is not available. Available methods for the lineal model are "Adam", "trust-constr" and "SLSQP"'
            )
        
        if self.link == 'log-lineal'and method.lower() != 'adam':
            raise ValueError(
                'The selected method is not available. Only "Adam" method is available for the log-lineal model'
            )

        self.method = method

        self.tol = tol
        self.num_iter = num_iter
        self.verbose = verbose
    
    def __call__(self):
        """
        Method to make the object callable that executes the fitting, prediction and evaluation
        of the CV metric through a grid of values for the order of the parameters p and q and
        returns the model with the best metric
        """

        cv_scores = []

        for (p,q) in self.grid:

            model = PoissonAutoregression(p=p, q=q, link=self.link)
            cv_score = ts_cv_score(model=model, series=self.series, method=self.method, tol=self.tol, num_iter=self.num_iter)
            cv_scores.append(cv_score)

            if self.verbose:
                print(f'Model fited with p = {p} and q = {q} returns a cv_score of {cv_score}')
            

        self.cv_scores = np.array(cv_scores)
        self.p = self.grid[self.cv_scores.argmin()][0]
        self.q = self.grid[self.cv_scores.argmin()][1]

        print(f'The best model uses parameters p = {self.p} and q = {self.q} with a cv_score of {self.cv_scores.min()}')

        best_model = PoissonAutoregression(p=self.p,q=self.q, link=self.link)
        
        self.best_model = best_model

        return best_model
    

class MultivariatePoissonAutorregresion():
    """
    Class defining a multivariate forcaster that implements Poisson
    autoregressive (also called INGARCH) models on each series on the data set.
    """
    def __init__(self,
        max_p:int=12,
        max_q:int=12,
        cv_folds:int=5,
        link:str="log-lineal",
        method:str="Adam"
    ):
        """
        Class instantiation method.

        Args
        ----------
            max_p:int=12
                Maximum autorregresive order for the values
            max_q:int=12
                Maximum autorregresive order for the errors
            cv_folds:int=5
                Number of train-test splits to be used in the cross-validation
            link:str="log-lineal"
                Link function for the mean and the systematic component. Can be
                "lineal" or "log-lineal"
            method:str="Adam"
                Method used to find the parameters optimal values. For the log-lineal model
                only the Adam method is available.

                The available methods for the lineal model are "Adam" gradient descent and 
                "trust-constr" or "SLSQP" for the implementation in scipy 
                (see https://docs.scipy.org/doc/scipy/tutorial/optimize.html#constrained-minimization-of-multivariate-scalar-functions-minimize)
                for the Trust-Region Constrained Algorithm and the Sequential Least Squares Programming Algorithm
            
                Note that the "Adam" gradient descent is an unconstrained method and may fall to
                find values that fulfill the condition to garantee stationarity for the series, in which case
                the warning will shown.
        """
        self.max_p=max_p
        self.max_q=max_q
        self.cv_folds=cv_folds
        self.link=link
        self.method=method

    def fit(self, data:np.ndarray):
        """
        Method for the automatic instantiation and automatic autorregresive order
        selection for the models for each series in the data using the AutoINGARCH
        class.
        
        Args:
        ----------
            data:np.ndarray
                2D array containing the time series, with each searies as a column
        """
        cols = data.shape[1]
        self.models = {}
        self.fitted_values=[]

        for i in range(cols):
            auto_ingarch = AutoINGARCH(
                series=data[:,i], 
                max_p=self.max_p, 
                max_q=self.max_q, 
                cv_folds=self.cv_folds, 
                num_iter=100
            )
            ingarch_model = auto_ingarch()
            self.models["poisson_autoregression_"+str(i)] = PoissonAutoregression(
                p=ingarch_model.p,
                q=ingarch_model.q,
                link="log-lineal"
            )
            self.models["poisson_autoregression_"+str(i)].fit(data[:,i])
            self.fitted_values.append(self.models["poisson_autoregression_"+str(i)].fitted)

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