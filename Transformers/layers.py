"""
Module containing the definition for all layers required for
the transformer model
"""
import tensorflow as tf
import numpy as np

def positional_encoding(lenght:int, depth:int):
    """
    Function that calculates the positional encoding for a 
    given array lenght and depth

    Args:
    ----------
        lenght:int
            lenght of the positional encoding array
        depth:int
            Dimension of each sample in the array

    Returns:
    ---------
        np.array
            Array containing the sinusoidal frecuencies
            corresponding to the positional encoding.
    """

    depth = depth/2 

    positions = np.arange(lenght)[:, np.newaxis] 
    depths = np.arange(depth)[np.newaxis, :]/depth 

    angle_rates = 1/(10000**depths)
    angle_rads = positions*angle_rates

    pos_encoding = np.concatenate(
        [np.sin(angle_rads), np.cos(angle_rads)],
        axis=-1
    )

    return tf.cast(pos_encoding, dtype=tf.float32)

class PositionalEncoder(tf.keras.layers.Layer):
    """
    Layer that gives a positional encoding to the input
    to ensure that information about the position of each 
    element in the sequence is past to the model
    """
    def __init__(self, d_model:int, lenght:int):
        """
        Layer's instantiation method.

        Args:
        ---------
            d_model:int
                Dimension of each element in the secuence within 
                the model. For this architecture it is the same as 
                both the encoding and decoding inputs dimension
            lenght:int
                Lenght of the sequences past in each batch
        -------
        """
        super().__init__()
        self.d_model = d_model
        self.pos_encoding = positional_encoding(lenght=lenght, depth=d_model)

    def call(self, x):
        """
        Method to calculate the layer's output

        Args:
        ---------
            x:tensor
                Tensor containing the layer's inputs
        """
        length = tf.shape(x)[1]
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = x + self.pos_encoding[tf.newaxis, :length, :]
        return x
    
class BaseAttention(tf.keras.layers.Layer):
    """
    Generic class for attention layers
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()

    def call(self, **kwargs):
        return NotImplementedError
    

class CrossAttention(BaseAttention):
    """
    Class containing the attention algorithm that queries
    the decoder's self attention outputs againts the 
    encoder's outputs (the context)
    """
    def call(self, x, context):
        attn_output, attn_scores = self.mha(
            query=x,
            key=context, 
            value=context,
            return_attention_scores=True
        )

        # Cache the attention scores for plotting later.
        self.last_attn_scores = attn_scores

        x = self.add([x, attn_output])
        x = self.layernorm(x)

        return x
    

class GlobalSelfAttention(BaseAttention):
    """
    Class with the self attention algorithm applied in the
    encoder, where all elements in the sequence can be queried
    globally for each element.
    """
    def call(self, x):
        attn_output = self.mha(
            query=x,
            value=x,
            key=x
        )
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x
    

class CausalSelfAttention(BaseAttention):
    """
    Class with the self attention algorithm applied in the
    decoder, where each element can only query those previous
    to it, to maintain the autorregresive structure of the model,
    wich requires a causal mask for all inputs after each element.
    """
    def call(self, x):
        attn_output = self.mha(
            query=x,
            value=x,
            key=x,
            use_causal_mask = True
        )
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x
    

class FeedForward(tf.keras.layers.Layer):
    """
    Class defyning a fully conected neural network to process the
    outputs of the attention alforithms in both the encoder and
    decoder.
    """
    def __init__(self, d_model:int, dff:int, dropout_rate:float=0.1):
        """
        The class instantiation method

        Args:
        ----------
            d_model:int
                Dimension of each element in the secuence within 
                the model. For this architecture it is the same as 
                both the encoding and decoding inputs dimension
            dff:int
                Number of units to be used in the first layer
                of the fully connected network
            dropout_rate:float=0.1
        """
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(dff, activation='relu')
        self.dense2 = tf.keras.layers.Dense(d_model)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.add = tf.keras.layers.Add()
        self.layernorm = tf.keras.layers.LayerNormalization()

    def call(self, x):
        y = self.dense1(x)
        y = self.dense2(y)
        y = self.dropout(y)
        x = self.add([x,y])
        x = self.layernorm(x)
        return x
    

class EncoderLayer(tf.keras.layers.Layer):
    """
    Class with the base structure for the encoder layer
    """
    def __init__(
            self, d_model:int, num_heads:int, dff:int, dropout_rate=0.1
        ):
        """
        The class instantiation method

        Args:
        ----------
            d_model:int
                Dimension of each element in the secuence within 
                the model. For this architecture it is the same as 
                both the encoding and decoding inputs dimension
            num_heads:int
                Number of attention heads on each multihead attention
                applied in the model
            dff:int
                Number of units to be used in the first layer
                of the fully connected network
            dropout_rate:float=0.1
        """
        super().__init__()

        self.self_attention = GlobalSelfAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate
        )

        self.ffn = FeedForward(d_model, dff)

    def call(self, x):
        x = self.self_attention(x)
        x = self.ffn(x)
        return x
    

class Encoder(tf.keras.layers.Layer):
    """
    Class defining the model's full encoder
    """
    def __init__(
        self, 
        num_layers:int, 
        d_model:int,
        lenght:int,
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
            num_heads:int
                Number of attention heads on each multihead attention
                applied in the model
            dff:int
                Number of units to be used in the first layer
                of the fully connected network
            dropout_rate:float=0.1
        """
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.pos_embedding = PositionalEncoder(d_model=d_model, lenght=lenght)

        self.enc_layers = [
            EncoderLayer(
                d_model, num_heads, dff, dropout_rate
            ) for _ in range(num_layers)
        ]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x):
        # 'x' shape: (batch_size, seq_len, d_model)
        x = self.pos_embedding(x) 
        # Add dropout.
        x = self.dropout(x)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x)

        return x 
    

class DecoderLayer(tf.keras.layers.Layer):
    """
    Class with the base structure for the decoder layer
    """
    def __init__(
            self, d_model:int, num_heads:int, dff:int, dropout_rate:float=0.1
        ):        
        """
        The class instantiation method

        Args:
        ----------         
            d_model:int
                Dimension of each element in the secuence within 
                the model. For this architecture it is the same as 
                both the encoding and decoding inputs dimension
            num_heads:int
                Number of attention heads on each multihead attention
                applied in the model
            dff:int
                Number of units to be used in the first layer
                of the fully connected network
            dropout_rate:float=0.1
        """
        super(DecoderLayer, self).__init__()

        self.causal_self_attention = CausalSelfAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate
        )

        self.cross_attention = CrossAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate
        )

        self.ffn = FeedForward(
            d_model=d_model,
            dff=dff
        )

    def call(self, x, context):
        x = self.causal_self_attention(x=x)
        x = self.cross_attention(x=x, context=context)

        # Cache the last attention scores for plotting
        self.last_attn_scores = self.cross_attention.last_attn_scores

        x = self.ffn(x) # shape: (batch_size, seq_len, d_model)
        return x


class Decoder(tf.keras.layers.Layer):
    """
    Class defining the model's full decoder
    """
    def __init__(self,
        num_layers:int, 
        d_model:int,
        lenght:int,
        num_heads:int, 
        dff:int, 
        dropout_rate:float=0.1, 
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
            num_heads:int
                Number of attention heads on each multihead attention
                applied in the model
            dff:int
                Number of units to be used in the first layer
                of the fully connected network
            dropout_rate:float=0.1
        """
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.pos_embedding = PositionalEncoder(
            d_model=d_model, lenght=lenght
        )
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.dec_layers = [
            DecoderLayer(
                d_model=d_model, num_heads=num_heads, dff=dff, dropout_rate=dropout_rate
            ) for _ in range(num_layers)
        ]

        self.last_attn_scores = None

    def call(self, x, context):

        # x shape: (batch, target_seq_len, d_model)
        x = self.pos_embedding(x) 

        x = self.dropout(x)

        for i in range(self.num_layers):
            x = self.dec_layers[i](x, context)

        self.last_attn_scores = self.dec_layers[-1].last_attn_scores

        return x