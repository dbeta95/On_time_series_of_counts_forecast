import os
import sys

import numpy as np
import tensorflow as tf

from typing import Optional

home_path = os.path.dirname(os.path.dirname(__file__))
sys.path.append(home_path)

from Optimization.optimizers import Adam

class PoissonRegression():

    def __init__(self, epsilon:Optional[float]=1e-8):
        """
        Class instantiation method
        """
        self.epsilon = epsilon

    def __loss_function__(self,
        X:np.ndarray,
        y:np.ndarray,
        w:np.ndarray,
        epsilon:float=1e-8    
    ):
        """
        Functión to estimate the loss function given by the negative log-likelihood of the
        log-linear Poisson Regression model

        Args
        ----------
            X:np.ndarray
                Matrix of the observed values for the regression variables
            y:np.ndarray
                Vector of the observed values for the random component wich follows a Poisson distribution    
            w:np.ndarray
                vector of weights for each variable in the systematic component
            epsilon:float=1e-8
                Correction factor to avoid math errors
        """
        y_hat = 1/(np.exp(-(X @ w))+epsilon)
        error = (y_hat - np.log(y_hat) * y).sum()
        return error

    def __grad__(self,
        X:np.ndarray,
        y:np.ndarray,
        w:np.ndarray,
        epsilon:float=1e-8
    ):
        """
        Functión to get the gradient for the parameters vector and biass

        Args
        ----------
            X:np.ndarray
                Matrix of the observed values for the regression variables
            y:np.ndarray
                Vector of the observed values for the random component wich follows a Poisson distribution    
            w:np.ndarray
                vector of weights for each variable in the systematic component
            epsilon:float=1e-8
                Correction factor to avoid math errors
        """
        y_hat = 1/(np.exp(-(X @ w))+epsilon)
        dw = (X.T @ (y_hat - y))
        return dw

    def fit(self,
        X:np.ndarray,
        y:np.ndarray,
        w_0:np.ndarray=None, 
        alpha:float=0.01, 
        beta_1:float=0.9, 
        beta_2:float=0.99, 
        epsilon:float=1e-8,
        tol:float=1e-6,
        num_iter:int=10000
    ):
        """
        Method to estimate the parameters values by solving the minimize problem
        using the adam algorithm for both the weights vector and the bias.

        Args:
        ----------
            X:np.ndarray
                Matrix of the observed values for the regression variables
            y:np.ndarray
                Vector of the observed values for the random component wich follows a Poisson distribution  
            w:np.ndarray
                vector of weights for each variable in the systematic component
            alpha: float
                Learning rate       
            beta_1: float.
                First moment decay            
            beta_2: float.
                Seconds moment decay           
            epsilon: float.
                Stabilizing factor
            tol: float
                Bound of change to stop the algorithm 
            num_iter:int
                Maximum number of iterations.
        """
        X = np.concatenate((np.ones(X.shape[0]).reshape(-1,1), X), axis = 1)
        p = X.shape[1]
        if w_0 is None:
            w_0 = np.zeros(p)
        w = w_0.copy()
        m = np.zeros(p)
        v = np.zeros(p)

        self.hist = np.zeros(num_iter)

        gradient_descent = Adam(
            alpha = alpha, 
            beta_1 = beta_1, 
            beta_2 = beta_2, 
            epsilon = epsilon,
            m = m,
            v = v
        )

        for iter in range(num_iter):

            dw = self.__grad__(X, y, w)
            w = gradient_descent.update(theta=w, grad_theta=dw)

            self.hist[iter] = self.__loss_function__(X, y, w)

            if np.abs(self.hist[iter]-self.hist[iter-1]) < tol:
                break

        self.coefs = w[1:]
        self.intercept = w[0]

    def predict(self,X:np.ndarray, as_integers:bool=False,epsilon:float=1e-8):
        """
        Method to predict the response values for a given matrix X

        Args:
        ----------
            X:np.ndarray
                Matrix of the observed values for the regression variables
            as_integers:bool
                Should the valuesbe presented rounded as integers
            epsilon: float.
                Stabilizing factor
        Results:
        ----------
            y_hat:np.ndarray
                Vactor of predicted values for the response variable
        """
        y_hat = 1./(np.exp(- (X @ self.coefs + self.intercept))+epsilon)
        if as_integers:
            y_hat = np.round(y_hat)
        return y_hat
    
class PoissonRegressionSyntheticData():
    """
    Module to generate synthetic data for lineal regression
    """
    def __init__(self, 
        w:tf.Tensor, b:tf.Tensor, 
        num_train:int=1000,num_val:int=200,batch_size:int=32,
        seed:int=None
    ):
        """
        Class initialization method
        """
        self.num_train = num_train
        self.num_val = num_val
        self.batch_size = batch_size
        self.n = num_train + num_val
        if seed:
            tf.random.set_seed(seed)            
        self.seed = seed
        x1 = tf.random.uniform((self.n, 1), seed=1)
        x2 = tf.random.normal((self.n,1), seed=1)
        self.X = tf.concat([x1,x2], axis=1)
        lmbd = tf.matmul(self.X, tf.reshape(w, (-1,1))) + b
        self.y = tf.reshape(tf.random.poisson((1,), tf.exp(lmbd),seed=1),(-1,1))

    def training_loader(self):
        """
        Method to create the data loader for the training process
        """
        dataset = tf.data.Dataset.from_tensor_slices((self.X[:self.num_train],self.y[:self.num_train]))
        return dataset.shuffle(buffer_size=dataset.cardinality(), seed=self.seed).batch(self.batch_size)
    
    def validation_loader(self):
        """
        Method to create the data loader for the validation process
        """
        dataset = tf.data.Dataset.from_tensor_slices((self.X[self.num_train:],self.y[self.num_train:]))
        return dataset.batch(self.batch_size)
    
class PossionRegressionNN(tf.keras.Model):

    def __init__(self, input_shape):
        super(PossionRegressionNN, self).__init__()

        self.dense = tf.keras.layers.Dense(
            1, input_shape=input_shape, activation=tf.exp
        )

    def call(self, input):
        return self.dense(input)