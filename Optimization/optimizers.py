"""
Module for self defined classes and funtions for optimization
"""
import numpy as np

from typing import Optional

class Adam:
    """
    Class to define the implementation of the Adam gradient descent as 
    explained in 
    https://towardsdatascience.com/understanding-gradient-descent-and-adam-optimization-472ae8a78c10
    """

    def __init__(self,         
        alpha:Optional[float] = 0.01,
        beta_1:Optional[float] = 0.9,
        beta_2:Optional[float] = 0.99,
        epsilon:Optional[float] = 1e-8,
        m:Optional[np.ndarray] = None,
        v:Optional[np.ndarray] = None
    ):
        """
        Class instantiation method

        Args:       
            alpha:Optional[float] = 0.01
                Learning rate
            beta_1:Optional[float] = 0.9
                Exponential decay rate for the first moment
            beta_2:Optional[float] = 0.99
                Exponential decay rate for the second moment
            epsilon:Optional[float] = 1e-8
                Parameter for numérical stability
            m:np.ndarray
                Initial value for the first moment vector
            v:np.ndarray
                Initial value for the second moment vector   
        """
        self.alpha = alpha
        self.beta_1 = beta_1
        self.beta_2 = beta_2

        self.m = m
        self.v = v
        self.t = 0

        self.epsilon = epsilon

    def update(self,
        theta:np.ndarray,
        grad_theta:np.ndarray
    ):
        """
        Parameters update method

        Args:
        ----------
        theta:np.ndarray
            Parameters to optimize in the previous step

        grad_theta:np.ndarray
            Loss functions gradient with respect to the parameters in the previous step

        Returns:
        ----------
        np.ndarray
            Actualized parameters
        """
        if self.m is None:
            self.m = np.zeros_like(grad_theta)
        
        if self.v is None:
            self.v = np.zeros_like(grad_theta)

        self.t += 1

        self.m = self.beta_1*self.m + (1. - self.beta_1)*grad_theta
        self.v = self.beta_2*self.v + (1. - self.beta_2)*grad_theta**2

        m_corrected = self.m / (1. - np.power(self.beta_1, self.t))
        v_corrected = self.v / (1. - np.power(self.beta_2, self.t))

        theta = theta - self.alpha * (m_corrected / (np.sqrt(v_corrected)+self.epsilon))

        return theta