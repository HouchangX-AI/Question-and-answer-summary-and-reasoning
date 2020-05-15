import tensorflow as tf


class BahdanauAttentionCoverage(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttentionCoverage, self).__init__()
        self.Wc = tf.keras.layers.Dense(units)
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, dec_hidden, enc_output, enc_padding_mask, use_coverage=False, prev_coverage=None):
        """
        :param dec_hidden: shape=(16, 256)
        :param enc_output: shape=(16, 200, 256)
        :param enc_padding_mask: shape=(16, 200)
        :param use_coverage:
        :param prev_coverage: None
        :return:
        """
        # hidden shape == (batch_size, hidden size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        # we are doing this to perform addition to calculate the score
        hidden_with_time_axis = tf.expand_dims(dec_hidden, 1)  # shape=(16, 1, 256)
        # att_features = self.W1(enc_output) + self.W2(hidden_with_time_axis)

        def masked_attention(score):
            """
            :param score: shape=(16, 200, 1)
                        ...
              [-0.50474256]
              [-0.47997713]
              [-0.42284346]]]
            :return:
            """
            attn_dist = tf.squeeze(score, axis=2)  # shape=(16, 200)
            attn_dist = tf.nn.softmax(attn_dist, axis=1)  # shape=(16, 200)
            mask = tf.cast(enc_padding_mask, dtype=attn_dist.dtype)
            attn_dist *= mask
            masked_sums = tf.reduce_sum(attn_dist, axis=1)
            attn_dist = attn_dist / tf.reshape(masked_sums, [-1, 1])
            attn_dist = tf.expand_dims(attn_dist, axis=2)
            return attn_dist

        if use_coverage and prev_coverage is not None:  # non-first step of coverage
            # Multiply coverage vector by w_c to get coverage_features.
            # Calculate v^T tanh(W_h h_i + W_s s_t + w_c c_i^t + b_attn)
            # shape (batch_size,attn_length)
            e = self.V(tf.nn.tanh(self.W1(enc_output) + self.W2(hidden_with_time_axis) + self.Wc(prev_coverage)))
            # Calculate attention distribution
            attn_dist = masked_attention(e)
            # Update coverage vector
            coverage = attn_dist + prev_coverage

        else:
            # Calculate v^T tanh(W_h h_i + W_s s_t + b_attn)
            e = self.V(tf.nn.tanh(self.W1(enc_output) + self.W2(hidden_with_time_axis)))  # shape=(16, 200, 1)
            # Calculate attention distribution
            attn_dist = masked_attention(e)  # shape=(16, 200, 1)
            if use_coverage:  # first step of training
                coverage = attn_dist  # initialize coverage
            else:
                coverage = []

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attn_dist * enc_output  # shape=(16, 200, 256)
        context_vector = tf.reduce_sum(context_vector, axis=1)  # shape=(16, 256)
        # coverage  shape=(16, 200, 1)
        return context_vector, tf.squeeze(attn_dist, -1), coverage


class Decoder(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz, embedding_matrix):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                                   weights=[embedding_matrix],
                                                   trainable=False)
        self.gru = tf.keras.layers.GRU(self.dec_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        # self.dropout = tf.keras.layers.Dropout(0.5)
        self.fc = tf.keras.layers.Dense(vocab_size, activation=tf.keras.activations.softmax)
        # self.fc = tf.keras.layers.Dense(vocab_size)

    def call(self, x, hidden, enc_output, context_vector):
        # def call(self, x, context_vector):

        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)

        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # passing the concatenated vector to the GRU
        output, state = self.gru(x)
        # output shape == (batch_size * 1, hidden_size)
        output = tf.reshape(output, (-1, output.shape[2]))

        # output shape == (batch_size, vocab)
        # out = self.dropout(output)
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