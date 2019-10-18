import numpy as np
import tensorflow as tf


class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)

        gru = tf.keras.layers.GRU(self.enc_units,
                                  return_sequences=True,
                                  return_state=True,
                                  recurrent_initializer='glorot_uniform')
        self.bigru = tf.keras.layers.Bidirectional(gru, merge_mode='concat')

    def call(self, x, hidden):
        x = self.embedding(x)
        hidden = tf.split(hidden, num_or_size_splits=2, axis=1)
        output, forward_state, backward_state = self.bigru(x, initial_state=hidden)
        state = tf.concat([forward_state, backward_state], axis=1)
        return output, state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, 2*self.enc_units))


class BahdanauAttention(tf.keras.Model):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        # hidden shape == (batch_size, hidden size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        # we are doing this to perform addition to calculate the score
        hidden_with_time_axis = tf.expand_dims(query, 1)

        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units)
        score = self.V(tf.nn.tanh(self.W1(values) + self.W2(hidden_with_time_axis)))
        print('aaaaa is {}'.format(self.V))

        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)

        gru = tf.keras.layers.GRU(self.dec_units,
                                  return_sequences=True,
                                  return_state=True,
                                  recurrent_initializer='glorot_uniform')
        self.bigru = tf.keras.layers.Bidirectional(gru, merge_mode='concat')
        self.fc = tf.keras.layers.Dense(vocab_size)
        self.fc = tf.nn.dropout(0.5)

        # used for attention
        self.attention = BahdanauAttention(self.dec_units)

    def call(self, x, hidden, enc_output):
        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)

        # enc_output shape == (batch_size, max_length, hidden_size)
        context_vector, attention_weights = self.attention(hidden, enc_output)

        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # passing the concatenated vector to the GRU
        hidden = tf.split(hidden, num_or_size_splits=2, axis=1) # kaushik added this line
        # output, state = self.bigru(x) # kaushik- not sure if we need hidden below???
        output, forward_state, backward_state = self.bigru(x, initial_state=hidden)
        state = tf.concat([forward_state, backward_state], axis=1)

        # output shape == (batch_size * 1, hidden_size)
        output = tf.reshape(output, (-1, output.shape[2]))

        # output shape == (batch_size, vocab)
        x = self.fc(output)

        return x, state, attention_weights


if __name__ == '__main__':
    encoder = Encoder(vocab_size=25216, embedding_dim=256, enc_units=1024, batch_sz=64)
    sample_hidden = encoder.initialize_hidden_state()
    example_input_batch = tf.ones(shape=(64, 88), dtype=tf.int32)
    sample_output, sample_hidden = encoder(example_input_batch, sample_hidden)
    print('Encoder output shape: (batch size, sequence length, units) {}'.format(sample_output.shape))
    print('Encoder Hidden state shape: (batch size, units) {}'.format(sample_hidden.shape))

    attention_layer = BahdanauAttention(128)
    attention_weights, attention_result = attention_layer(sample_hidden, sample_output)
    print("Attention result shape: (batch size, units) {}".format(attention_result.shape))
    print("Attention weights shape: (batch_size, sequence_length, 1) {}".format(attention_weights.shape))

    decoder = Decoder(vocab_size=13053, embedding_dim=256, dec_units=1024, batch_sz=64)
    sample_decoder_output, _, _ = decoder(tf.random.uniform((64, 1)), sample_hidden, sample_output)
    print('Decoder output shape: (batch_size, vocab size) {}'.format(sample_decoder_output.shape))


class Pointer(tf.keras.layers.Layer):

    def __init__(self):
        super(Pointer, self).__init__()
        self.w_s_reduce = tf.keras.layers.Dense(1)
        self.w_i_reduce = tf.keras.layers.Dense(1)
        self.w_c_reduce = tf.keras.layers.Dense(1)

    def call(self, context_vector, state, dec_inp):
        return tf.nn.sigmoid(self.w_s_reduce(state) + self.w_c_reduce(context_vector) + self.w_i_reduce(dec_inp))

optimizer = tf.keras.optimizers.Adagrad(params['learning_rate'],
                                        initial_accumulator_value=params['adagrad_init_acc'],
                                        clipnorm=params['max_grad_norm'])


