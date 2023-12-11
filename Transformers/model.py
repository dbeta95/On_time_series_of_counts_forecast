"""
Module containing the full model with the architechture 
implemented in this work.
"""
import tensorflow as tf
from Transformers.layers import Encoder, Decoder

class Transformer(tf.keras.Model):
    """
    Class with the full transformer model as implemented in this
    work
    """
    def __init__(self,
        num_layers:int, 
        d_model:int, 
        lenght:int,
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
            lenght:int
                lenght of the positional encoding array
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
            lenght=lenght
        )
        self.decoder = Decoder(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            dff=dff,
            dropout_rate=dropout_rate,
            lenght=lenght
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