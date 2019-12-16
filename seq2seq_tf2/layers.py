import tensorflow as tf
from seq2seq_tf2 import config
from utils.data_utils import load_word2vec


class Encoder(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz, embedding_matrix):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        # self.enc_units = enc_units
        self.enc_units = enc_units // 2
        self.embedding = tf.keras.layers.Embedding(vocab_size,
                                                   embedding_dim,
                                                   weights=[embedding_matrix],
                                                   trainable=False)
        gpus = tf.config.experimental.list_physical_devices('GPU')
        print('gpus number is ', gpus)
        if gpus:
            self.gru = tf.keras.layers.CuDNNGRU(self.enc_units,
                                                return_sequences=True,
                                                return_state=True,
                                                recurrent_initializer='glorot_uniform')
        else:
            self.gru = tf.keras.layers.GRU(self.enc_units,
                                           return_sequences=True,
                                           return_state=True,
                                           recurrent_initializer='glorot_uniform')
            self.bigru = tf.keras.layers.Bidirectional(self.gru, merge_mode='concat')

    def call(self, x, hidden):
        x = self.embedding(x)

        hidden = tf.split(hidden, num_or_size_splits=2, axis=1)
        output, forward_state, backward_state = self.bigru(x, initial_state=hidden)
        state = tf.concat([forward_state, backward_state], axis=1)
        # output, state = self.gru(x, initial_state=hidden)
        return output, state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, 2*self.enc_units))


class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, params):
        super(BahdanauAttention, self).__init__()
        self.params = params
        if self.params["is_coverage"]:
            self.Wc = tf.keras.layers.Dense(self.params["attn_units"])
        self.W1 = tf.keras.layers.Dense(self.params["attn_units"])
        self.W2 = tf.keras.layers.Dense(self.params["attn_units"])
        self.V = tf.keras.layers.Dense(1)

    def call(self, dec_hidden, enc_output, enc_padding_mask, coverage=0):
        """
        dec_hidden:
        """
        # hidden shape == (batch_size, hidden size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        # we are doing this to perform addition to calculate the score
        hidden_with_time_axis = tf.expand_dims(dec_hidden, 1)
        # att_features = self.W1(enc_output) + self.W2(hidden_with_time_axis)

        def masked_attention(e):
            """Take softmax of e then apply enc_padding_mask and re-normalize"""
            attn_dist = tf.nn.softmax(e)  # take softmax. shape (batch_size, attn_length)
            attn_dist *= enc_padding_mask  # apply mask
            masked_sums = tf.reduce_sum(attn_dist, axis=1)  # shape (batch_size)
            return attn_dist / tf.reshape(masked_sums, [-1, 1])  # re-normalize

        if self.params["is_coverage"] and coverage is not None:  # non-first step of coverage
            # Multiply coverage vector by w_c to get coverage_features.
            # Calculate v^T tanh(W_h h_i + W_s s_t + w_c c_i^t + b_attn)
            e = tf.reduce_sum(self.V(tf.nn.tanh(self.W_s(enc_output) +
                                                self.W_h(hidden_with_time_axis) +
                                                self.W_c(coverage))))  # shape (batch_size,attn_length)

            # Calculate attention distribution
            attn_dist = masked_attention(e)

            # Update coverage vector
            coverage += attn_dist

        else:
            # Calculate v^T tanh(W_h h_i + W_s s_t + b_attn)
            e = tf.reduce_sum(self.V(tf.nn.tanh(self.W_s(enc_output) + self.W_h(hidden_with_time_axis))))

            # Calculate attention distribution
            attn_dist = masked_attention(e)

            if self.params["is_coverage"]:  # first step of training
                coverage = tf.expand_dims(tf.expand_dims(attn_dist, 2), 2)  # initialize coverage

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attn_dist * enc_output
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attn_dist, coverage


class Decoder(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz, embedding_matrix):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                                   weights=[embedding_matrix],
                                                   trainable=False)
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            self.gru = tf.keras.layers.CuDNNGRU(self.enc_units,
                                                return_sequences=True,
                                                return_state=True,
                                                recurrent_initializer='glorot_uniform')
        else:
            self.gru = tf.keras.layers.GRU(self.dec_units,
                                           return_sequences=True,
                                           return_state=True,
                                           recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dropout(0.5)
        self.fc = tf.keras.layers.Dense(vocab_size, activation=tf.keras.activations.softmax)

    # def call(self, x, hidden, enc_output, context_vector):
    def call(self, x, context_vector):

        # enc_output shape == (batch_size, max_length, hidden_size)

        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)

        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # passing the concatenated vector to the GRU
        output, state = self.gru(x)
        # output shape == (batch_size * 1, hidden_size)
        output = tf.reshape(output, (-1, output.shape[2]))

        # output shape == (batch_size, vocab)
        out = self.fc(output)

        return x, out, state


class Pointer(tf.keras.layers.Layer):

    def __init__(self):
        super(Pointer, self).__init__()
        self.w_s_reduce = tf.keras.layers.Dense(1)
        self.w_i_reduce = tf.keras.layers.Dense(1)
        self.w_c_reduce = tf.keras.layers.Dense(1)

    def call(self, context_vector, state, dec_inp):
        return tf.nn.sigmoid(self.w_s_reduce(state) + self.w_c_reduce(context_vector) + self.w_i_reduce(dec_inp))


if __name__ == '__main__':
    from utils.data_utils import load_pkl
    word2vec = load_pkl('../datasets/word2vec.txt')
    print(word2vec)