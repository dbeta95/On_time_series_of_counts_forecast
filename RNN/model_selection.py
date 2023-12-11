"""
Module with the functions for architechture and hyperparameters 
selection for the RNN model
"""

import os
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from itertools import product

from RNN.data_processing import window_dataset


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
    sequence_lenght:int=12, 
    output_lenght:int=12,
    batch_size:int=32
):
    """
    train_data:np.ndarray
        Training data
    val_data:np.ndarray
        Validation data
    sequence_lenght:int
        Lenght of the sequence to be passed as input
        in each windw
    output_lenght:int
        Lenght of the output of each window
    batch_size:int
        Number of windows to be passed on each iteration
    """
    train_ds = window_dataset(
        data=train_data,
        sequence_lenght=sequence_lenght,
        output_lenght=output_lenght,
        batch_size=batch_size,
        shift=1,
        shuffle=True
    )
    val_ds = window_dataset(
        data=val_data,
        sequence_lenght=sequence_lenght,
        output_lenght=output_lenght,
        batch_size=batch_size,
        shift=1,
        shuffle=False
    )

    return train_ds, val_ds

def get_model(
        num_features:int, 
        sequence_lenght:int, 
        output_lenght:int,
        lstm_units:int=8,
        architechture:str="type-4",
        learning_rate:float=0.001,
    ):
    """ 
    Function that creates and compiles a model with a given architecture and
    hyperparameters

    Args:
    ----------
        num_features:int
            Number of series in the dataset 
        sequence_lenght:int
            Lenght of the input's sequence 
        output_lenght:int
            lenght of the output's sequence
        lstm_units:int
            Number of units on each lstm cell
        architechture:str
            Type of architechture. Options are:
            - "type-1": 
                single RNN layer and each timestep plus the
                hidden state generates an output
            - "type-2":
                Two RNN layers and each timestep plus the
                hidden state generates an output
            - "type-3": 
                single RNN layer and the output is generated
                simultaneusly using the layer's states and output
                as intput.
            - "type-4":
                Two RNN layers and the output is generated
                simultaneusly using the layer's states and output
                as intput.
        learning_rate:float
            Learning rate for the parameters actualization at each step of 
            the gradient descend

    """
    assert architechture in (
        "type-1", "type-2", "type-3", "type-4"
    ), "Invalid architecture. Valid options are 'type-1', 'type-2', 'type-3' or 'type-4'."

    if architechture == "type-1":
    
        model = tf.keras.models.Sequential([
            tf.keras.layers.LSTM(
                units=lstm_units, 
                return_sequences=True, 
                input_shape=(sequence_lenght,num_features)
            ),
            tf.keras.layers.Dense(units=num_features)
        ])
        model.compile(
            loss=tf.keras.losses.MeanSquaredError(),
            optimizer=tf.keras.optimizers.Adam(learning_rate),
            metrics=[tf.keras.metrics.MeanAbsoluteError()]
        )

    if architechture == "type-2":
    
        model = tf.keras.models.Sequential([
            tf.keras.layers.LSTM(
                units=lstm_units, 
                return_sequences=True, 
                input_shape=(sequence_lenght,num_features)
            ),
            tf.keras.layers.LSTM(
                units=lstm_units, 
                return_sequences=True
            ),
            tf.keras.layers.Dense(units=num_features)
        ])
        model.compile(
            loss=tf.keras.losses.MeanSquaredError(),
            optimizer=tf.keras.optimizers.Adam(learning_rate),
            metrics=[tf.keras.metrics.MeanAbsoluteError()]
        )

    if architechture == "type-3":
    
        model = tf.keras.models.Sequential([
            tf.keras.layers.LSTM(
                units=lstm_units, 
                return_sequences=False, 
                input_shape=(sequence_lenght,num_features)
            ),
            tf.keras.layers.Dense(
                units=output_lenght*num_features,
                kernel_initializer=tf.initializers.zeros()
            ),
            tf.keras.layers.Reshape([output_lenght, num_features])            
        ])
        model.compile(
            loss=tf.keras.losses.MeanSquaredError(),
            optimizer=tf.keras.optimizers.Adam(learning_rate),
            metrics=[tf.keras.metrics.MeanAbsoluteError()]
        )

    if architechture == "type-4":
    
        model = tf.keras.models.Sequential([
            tf.keras.layers.LSTM(
                units=lstm_units, 
                return_sequences=True, 
                input_shape=(sequence_lenght,num_features)
            ),            
            tf.keras.layers.LSTM(
                units=lstm_units, 
                return_sequences=False
            ),
            tf.keras.layers.Dense(
                units=output_lenght*num_features,
                kernel_initializer=tf.initializers.zeros()
            ),
            tf.keras.layers.Reshape([output_lenght, num_features])            
        ])
        model.compile(
            loss=tf.keras.losses.MeanSquaredError(),
            optimizer=tf.keras.optimizers.Adam(learning_rate),
            metrics=[tf.keras.metrics.MeanAbsoluteError()]
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

class RNNHyperTune():
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
                window size for the model. defines the outputh lenght and either
                the full or half the input lenght, depending on the architechture
            parameters:dict
                dictionary of lists of values for each parameter        
        """
        self.num_features = num_features
        self.window_size = window_size
        parameters['architechture'] = ["type-1", "type-2", "type-3", "type-4"]
        self.params_list = get_param_grid(parameters)

    def __get_trained_model__(self,
        train_data:np.ndarray,
        val_data:np.ndarray,
        num_features:int, 
        sequence_lenght:int, 
        output_lenght:int,
        lstm_units:int,
        architechture:str,
        learning_rate:float,
        batch_size:int,
        epochs:int
    ):
        
        train_ds, val_ds = get_windowed_datasets(
            train_data, val_data, sequence_lenght, output_lenght, batch_size
        )

        model = get_model(
            num_features=num_features,
            sequence_lenght=sequence_lenght,
            output_lenght=output_lenght,
            lstm_units=lstm_units,
            architechture=architechture,
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
        num_features:int, 
        sequence_lenght:int, 
        output_lenght:int,
        lstm_units:int,
        architechture:str,
        learning_rate:float,
        batch_size:int,
        checkpoint_path:str,
        epochs:int
    ):
        train_ds, val_ds = get_windowed_datasets(
            train_data, val_data, sequence_lenght, output_lenght, batch_size
        )

        model = get_model(
            num_features, 
            sequence_lenght,
            output_lenght,
            lstm_units,
            architechture,
            learning_rate
        )

        checkpoint_path = os.path.join(checkpoint_path, 'rnn_checkpoint')
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
            num_features, 
            sequence_lenght,
            output_lenght,
            lstm_units,
            architechture,
            learning_rate
        )
        new_model.load_weights(checkpoint_path)

        return new_model, history
        
        
    def fit(self, data: np.ndarray, checkpoint_path:str, epochs:int=100):
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

        for parameters in self.params_list:
            if parameters['architechture'] in ("type-1", "type-2"):
                parameters['sequence_lenght'] = self.window_size
                parameters['output_lenght'] = self.window_size
            if parameters['architechture'] in ("type-3", "type-4"):
                parameters['sequence_lenght'] = 2*self.window_size
                parameters['output_lenght'] = self.window_size

            history = self.__get_trained_model__(
                train_data,
                val_data,
                self.num_features,                
                epochs=epochs,
                **parameters
            )

            val_losses.append(min(history.history['val_loss']))
            
        min_loss_index = val_losses.index(min(val_losses))
        self.best_params = self.params_list[min_loss_index]

        self.best_model, self.history = self.__train_best_model__(
            train_data,
            val_data,
            self.num_features,                
            epochs=epochs,
            checkpoint_path=checkpoint_path,
            **self.best_params
        )

        return self.best_model, scaler
