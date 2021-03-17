import tensorflow as tf

from r2pa.transformer.positional_encoding import positional_encoding
from r2pa.transformer.multi_head_attention import MultiHeadAttention
from r2pa.transformer.point_wise_feed_forward_network import point_wise_feed_forward_network


class Encoder(tf.keras.layers.Layer):
    """
    Mostly taken from https://blog.tensorflow.org/2019/05/transformer-chatbot-tutorial-with-tensorflow-2.html
    """

    def __init__(self, num_layers, num_attributes, d_model, num_heads, dff, maximum_position_encoding, rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        # self.pos_encoding = tf.keras.backend.one_hot(50 * [[i // 4 for i in range(maximum_position_encoding)]], num_classes=maximum_position_encoding // 4)
        # self.pos_encoding = tf.range(maximum_position_encoding, dtype=tf.float32) / (maximum_position_encoding * 10)
        self.pos_encoding = positional_encoding(maximum_position_encoding, self.d_model)

        self.enc_layers = [EncoderLayer(num_attributes, d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask, tape, return_input_embeddings, track_embeddings, track_attention):
        seq_len = tf.shape(x)[1]  # TODO: Correct this!

        # adding position encoding.
        # x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        # x = tf.concat([x, self.pos_encoding], axis=-1)
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)
        if tape is not None:
            tape.watch(x)

        attentions = []
        embeddings = [x] if return_input_embeddings else []
        for i in range(self.num_layers):
            x, a = self.enc_layers[i](x, training, mask, track_attention=track_attention)
            if track_attention:
                attentions.append(tf.convert_to_tensor(a))
            if track_embeddings:
                embeddings.append(x)
            if tape is not None:
                tape.watch(x)

        return x, attentions, embeddings  # (batch_size, input_seq_len, d_model)


class EncoderLayer(tf.keras.layers.Layer):
    """
    Mostly taken from https://blog.tensorflow.org/2019/05/transformer-chatbot-tutorial-with-tensorflow-2.html
    """

    def __init__(self, num_attributes, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.num_attributes = num_attributes
        # one multi head attention block with num_heads heads for each attribute
        self.mha = [MultiHeadAttention(d_model, num_heads) for _ in range(self.num_attributes)]
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = [tf.keras.layers.Dropout(rate) for _ in range(self.num_attributes)]
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask, track_attention=False):
        attn_output, attentions = [], []
        # (num_attributes, batch_size, input_seq_len, d_model)
        for i, mha, d in zip(range(self.num_attributes), self.mha, self.dropout1):
            # select attribute corresponding data
            x_ = x[:, i::self.num_attributes]
            # calculate attention and attention weights for selected attribute with all attributes as input
            # i.e. at which input position and attributes to look at to encode selected attribute
            att_o, att = mha(x_, x_, x, mask[:, i::self.num_attributes])  # x_ value and key, x query
            attn_output.append(d(att_o))
            if track_attention:
                attentions.append(att)

        attn_output = tf.math.reduce_sum(attn_output, axis=0)

        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2, attentions
