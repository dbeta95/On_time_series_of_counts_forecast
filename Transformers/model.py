"""
Module containing the full model with the architechture 
implemented in this work.
"""
import tensorflow as tf
import numpy as np
from Transformers.layers import Encoder, Decoder

class Transformer(tf.keras.Model):
    """
    Class with the full transformer model as implemented in this
    work
    """
    def __init__(self,
        num_layers:int, 
        d_model:int, 
        input_lenght:int,
        output_lenght:int,
        d_target:int, 
        num_heads:int, 
        dff:int, 
        dropout_rate:float=0.1
    ):
        """
        The class instantiation method

        Args:
        ----------
            num_layers:int
                Number of encoder's layers to be implemented
                in the encoder.            
            d_model:int
                Dimension of each element in the secuence within 
                the model. For this architecture it is the same as 
                both the encoding and decoding inputs dimension
            input_lenght:int
                lenght of the positional encoding array for the input,
            output_lenght:int
                lenght of the positional encoding array for the output,
            d_target:int
                Dimension of each element in the target's sequence
            num_heads:int
                Number of attention heads on each multihead attention
                applied in the model
            dff:int
                Number of units to be used in the first layer
                of the fully connected network
            dropout_rate:float=0.1
        """
        super().__init__()
        
        self.encoder = Encoder(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            dff=dff,
            dropout_rate=dropout_rate,
            lenght=input_lenght
        )
        self.decoder = Decoder(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            dff=dff,
            dropout_rate=dropout_rate,
            lenght=output_lenght
        )
        self.final_layer = tf.keras.layers.Dense(d_target)

    def call(self, inputs):
        # To use a keras model with .fit all inputs must be passed in the
        # first argument
        context, x = inputs

        context = self.encoder(context) # (batch_size, context_len, d_model)

        x = self.decoder(x, context) # (batch_size, target_len, d_model)

        # Final linear layer output
        x = self.final_layer(x) # (batch_size, target_len, target_dim)

        try:
            # Drop the keras mask, so it doesn't scale the losses/metrics.
            del x._keras_mask
        except AttributeError:
            pass

        # Return the final output and the attention weights.
        return x
    
class Forecaster(tf.Module):
    """
    Class to forecast an specific number of steps using the transformer model
    """
    def __init__(self, scaler, transformer):
        """
        Class instantiation method

        Args:
        ---------
            scaler:
                Scaler object trained for the transformer model
            transformer:
                Transformer model trained
        """
        self.scaler = scaler
        self.transformer = transformer

    def __call__(self, inputs:np.ndarray, steps:int):
        """
        Forecastign method for the transformer

        Args:
        ---------
            inputs:np.ndarray
                Arry containing the inputs for the forecasting
            steps:int
                Periods to be forecasted.
        """
        scaled_data = self.scaler.transform(inputs)
        start = scaled_data[-1].reshape((1,-1))
        scaled_data = scaled_data[np.newaxis]
        encoder_input = tf.constant(scaled_data)
        output_array = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        output_array = output_array.write(0, start)

        for t in range(steps):
            output = tf.transpose(output_array.stack(), [1,0,2])
            prediction = self.transformer((encoder_input, output), training=False)
            prediction = tf.reshape(prediction[:, -1, :], (1,-1))
            output_array = output_array.write(t+1, prediction)

        output = tf.transpose(output_array.stack(), [1,0,2])
        output = output.numpy()[0,1:,:]

        return self.scaler.inverse_transform(output)