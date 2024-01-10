"""
Module with the functions for architechture and hyperparameters 
selection for the Transformer model
"""

import os
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from itertools import product

from Transformers.data_processing import window_dataset_sequence_to_sequence
from Transformers.model import Transformer

def get_datasets(data:np.ndarray, val_prop:float=0.2):
    """
    Function to get the training and validation datasets from the data.csv

    Args:
    ---------
        data:np.ndarray
            Data frame containing the simulated train data
        val_prop:float=0.3
            Proportiong of time steps to be used for validation
    """
    val_size = int(val_prop*data.shape[0])
    train_data, val_data = (
        data[:-val_size], 
        data[-val_size:]
    )
    scaler = MinMaxScaler()
    train_data = scaler.fit_transform(train_data)
    val_data = scaler.transform(val_data)    
    
    return train_data, val_data, scaler

def get_windowed_datasets(
    train_data:np.ndarray,
    val_data:np.ndarray,
    input_lenght:int=12, 
    output_lenght:int=12,
    batch_size:int=32
):
    """
    train_data:np.ndarray
        Training data
    val_data:np.ndarray
        Validation data
    input_lenght:int
        Lenght of the sequence to be passed as input
        in each windw
    output_lenght:int
        Lenght of the output of each window
    batch_size:int
        Number of windows to be passed on each iteration
    """
    train_ds = window_dataset_sequence_to_sequence(
        data=train_data,
        sequence_lenght=input_lenght,
        output_lenght=output_lenght,
        batch_size=batch_size,
        shift=1,
        shuffle=True
    )
    val_ds = window_dataset_sequence_to_sequence(
        data=val_data,
        sequence_lenght=input_lenght,
        output_lenght=output_lenght,
        batch_size=batch_size,
        shift=1,
        shuffle=False
    )

    return train_ds, val_ds

def get_model(
        d_model:int, 
        d_target:int, 
        num_layers:int,
        num_heads:int,
        dff:int,
        input_lenght:int,
        output_lenght:int,
        dropout_rate:float=0.1,
        learning_rate:float=0.001,        
    ):
    """ 
    Function that creates and compiles a model with a given set of 
    hyperparameters.

    Args:
    ----------
        d_model:int
            Number of series in the dataset 
        d_target:int
            Number of series being forecasted 
        num_layers:int
            Number of encoding and decoding layers
        num_heads:int
            Number of attention heads
        dff:int
            Number of nodes in the feed forward network
        input_lenght:int
            lenght of the positional encoding array for the input,
        output_lenght:int
            lenght of the positional encoding array for the output,
        dropout_rate:float
            Percentage of parameters being turned to zero on each step
        learning_rate:float
            Learning rate for the parameters actualization at each step of 
            the gradient descend

    """    
   
    model = Transformer(
        d_model=d_model,
        d_target=d_target,        
        num_layers=num_layers,
        num_heads=num_heads,
        dff=dff,
        dropout_rate=dropout_rate,
        input_lenght=input_lenght,
        output_lenght=output_lenght
    )
    model.compile(
        loss=tf.keras.losses.MeanSquaredError(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=[tf.keras.metrics.MeanAbsolutePercentageError()]
    )

    return model


def get_param_grid(parameters:dict):
    """
    Function to convert a dictionary of lists of parameters values
    into a list of dictionaries of parameters.

    Args:
    ----------
        parameters:dict
            Dictionary containing a list of values for the parameter
            under each parameter's name as key
    Returns:
    ---------
        list
            List of dictionaries of parametersand it's value
    """
    params_list = [[{item[0]:v} for v in item[1]]for item in parameters.items()]
    return [
        {k:v for e in l for (k,v) in e.items()} 
        for l in list(product(*params_list))
    ]

class TransformerHyperTune():
    """
    Class to tune among various model architechtures and hyperparameters
    for the RNN.
    """
    def __init__(self, num_features:int, window_size:int, parameters:dict):
        """
        Mehtod instantiation function

        Args:
        ---------
            num_features:int
                Number of time series in the dataset
            window_size:int
                window size for the model. defines the outputh lenght and half 
                the input lenght.
            parameters:dict
                dictionary of lists of values for each parameter        
        """
        self.d_model = num_features
        self.d_target = num_features
        self.input_lenght = window_size
        self.output_lenght = window_size
        self.params_list = get_param_grid(parameters)

    def __get_trained_model__(self,
        train_data:np.ndarray,
        val_data:np.ndarray,
        d_model:int,
        d_target:int,        
        num_layers:int,
        num_heads:int,
        dff:int,
        dropout_rate:float,
        input_lenght:int,
        output_lenght:int,
        learning_rate:float,
        batch_size:int,
        epochs:int
    ):
                
        train_ds, val_ds = get_windowed_datasets(
            train_data, val_data, input_lenght, output_lenght, batch_size
        )

        model = get_model(
            d_model=d_model,
            d_target=d_target,        
            num_layers=num_layers,
            num_heads=num_heads,
            dff=dff,
            dropout_rate=dropout_rate,
            input_lenght=input_lenght,
            output_lenght=output_lenght,
            learning_rate=learning_rate
        )

        return model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            verbose=0
        )
    
    def __train_best_model__(self,
        train_data:np.ndarray,
        val_data:np.ndarray,
        d_model:int,
        d_target:int,        
        num_layers:int,
        num_heads:int,
        dff:int,
        dropout_rate:float,
        input_lenght:int,
        output_lenght:int,
        learning_rate:float,
        batch_size:int,
        epochs:int,
        checkpoint_path:str
    ):
        train_ds, val_ds = get_windowed_datasets(
            train_data, val_data, input_lenght, output_lenght, batch_size
        )

        model = get_model(
            d_model=d_model,
            d_target=d_target,        
            num_layers=num_layers,
            num_heads=num_heads,
            dff=dff,
            dropout_rate=dropout_rate,
            input_lenght=input_lenght,
            output_lenght=output_lenght,
            learning_rate=learning_rate
        )

        checkpoint_path = os.path.join(checkpoint_path, 'transformer_checkpoint')
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            save_weights_only=True,
            save_freq='epoch',
            monitor='val_loss',
            save_best_only=True,
            verbose=0,

        )

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=[checkpoint],
            verbose=0
        )

        new_model = get_model(
            d_model=d_model,
            d_target=d_target,        
            num_layers=num_layers,
            num_heads=num_heads,
            dff=dff,
            dropout_rate=dropout_rate,
            input_lenght=input_lenght,
            output_lenght=output_lenght,
            learning_rate=learning_rate
        )
        new_model.load_weights(checkpoint_path)

        return new_model, history
        
        
    def fit(self, data: np.ndarray, checkpoint_path:str, epochs:int=100, verbose=True):
        """
        Method for the automatic selection of hyperparameter values. The method
        re-trains and saves the best model.
        
        Args:
        ----------
            data:np.ndarray
            checkpoint_path:str
            epochs:int=100

        Returns:
        ---------
            tf.keras.Model
                Best model        
        """
        train_data, val_data, scaler = get_datasets(data)
        val_losses = []
        models = len(self.params_list)

        for i,parameters in enumerate(self.params_list):
            if verbose:
                print(f"Training model {i}/{models}.")
            
            history = self.__get_trained_model__(
                train_data,
                val_data,
                input_lenght=self.input_lenght,
                output_lenght=self.output_lenght,
                d_model=self.d_model,
                d_target=self.d_target,                
                epochs=epochs,
                **parameters
            )
            val_loss = min(history.history['val_loss'])
            if verbose:
                print(f"minimum {i} model's validation loss: {val_loss}.")

            val_losses.append(val_loss)
            
        min_loss_index = val_losses.index(min(val_losses))
        self.best_params = self.params_list[min_loss_index]
        if verbose:
                print(f"Best model's validation loss: {min(val_losses)}.")

        self.best_model, self.history = self.__train_best_model__(
            train_data,
            val_data,
            d_model=self.d_model,
            d_target=self.d_target,
            input_lenght=self.input_lenght,
            output_lenght=self.output_lenght,              
            epochs=epochs,
            checkpoint_path=checkpoint_path,
            **self.best_params
        )

        return self.best_model, scaler
