"""
Module with utilities to process time series data for predicting with RNN
models
"""

import numpy as np
import tensorflow as tf

def window_dataset(
    data:np.ndarray,    
    sequence_lenght:int,
    output_lenght:int,
    batch_size:int,
    shift:int,
    output_index:list = None,
    shuffle:bool = True,
):
    """
    Funcion to partitioned the data in windows to train an RNN model

    Args:
    ---------
        data:np.ndarray
            Array containing the time series data
        sequence_lenght:int
            Number of time steps in the input sequence
        output_lenght:int
            Number of time steps in the output sequence
        batch_size:int
            Number of windows per batch
        shift:int
            Number of timesteps between windows
        output_index:list = None
            Index of the response variable
        shuffle:bool = True
            Either to shuffle the windows or not

    Returns:
    ---------
        tf.data.Dataset
            Dataset that yields the window batches
    """
    # Saving the data dim in case of shuffle being required
    n = data.shape[0]

    # The full window has the size of the inputs sequence plus the outpus
    window_size = sequence_lenght + output_lenght
    
    # Convert the data to a tensorflowdataset
    ds = tf.data.Dataset.from_tensor_slices(data)

    # Create the window function shifing the data
    ds = ds.window(window_size, shift=shift, drop_remainder=True)

    # Flatten the dataset and batch into the winow size
    ds = ds.flat_map(lambda x: x.batch(window_size))
    
    if shuffle:
        ds = ds.shuffle(n)

    if output_index is not None:
        ds = ds.map(lambda x: (x[:-output_lenght], x[-output_lenght:, output_index]))
    else:
        ds = ds.map(lambda x: (x[:-output_lenght], x[-output_lenght:]))

    ds = ds.batch(batch_size).prefetch(1)

    return ds