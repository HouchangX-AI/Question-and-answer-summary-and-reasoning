import tensorflow as tf
from seq2seq_tf2.layers import Encoder, BahdanauAttention, Decoder, Pointer
from utils.data_utils import load_word2vec


class PGN(tf.keras.Model):
    def __init__(self, params):
        super(PGN, self).__init__()
        # self.embedding_matrix = load_word2vec(params["vocab_size"])
        self.params = params
        self.encoder = Encoder(params["vocab_size"], params["embed_size"], params["enc_units"], params["batch_size"])
        self.attention = BahdanauAttention(params["attn_units"])
        self.decoder = Decoder(params["vocab_size"], params["embed_size"], params["dec_units"], params["batch_size"])
        self.pointer = Pointer()

    def call_encoder(self, enc_inp):
        enc_hidden = self.encoder.initialize_hidden_state()
        enc_output, enc_hidden = self.encoder(enc_inp, enc_hidden)
        return enc_hidden, enc_output

    def call_decoder_onestep(self, latest_tokens, enc_hidden, dec_hidden):
        if self.params["pointer_gen"]:
            # TODO: implement
            pass
        else:
            context_vector, _ = self.attention(dec_hidden, enc_hidden)
            dec_x, pred, dec_hidden = self.decoder(latest_tokens,
                                                   None,
                                                   None,
                                                   context_vector)
        return dec_x, pred, dec_hidden

    def call(self, enc_output, dec_hidden, enc_inp, enc_extended_inp, dec_inp, batch_oov_len):
        predictions = []
        attentions = []
        p_gens = []
        context_vector, _ = self.attention(dec_hidden, enc_output)

        if self.params["pointer_gen"]:
            for t in range(dec_inp.shape[1]):
                dec_x, pred, dec_hidden = self.decoder(tf.expand_dims(dec_inp[:, t], 1),
                                                       dec_hidden,
                                                       enc_output,
                                                       context_vector)
                context_vector, attn = self.attention(dec_hidden, enc_output)
                p_gen = self.pointer(context_vector, dec_hidden, tf.squeeze(dec_x, axis=1))

                attentions.append(attn)
                p_gens.append(p_gen)
                predictions.append(pred)

            final_dists = _calc_final_dist(enc_extended_inp, predictions, attentions, p_gens, batch_oov_len,
                                           self.params["vocab_size"], self.params["batch_size"])

            if self.params["mode"] == "train":
                return tf.stack(final_dists, 1), dec_hidden  # predictions_shape = (batch_size, dec_len, vocab_size) with dec_len = 1 in pred mode
            else:
                return tf.stack(final_dists, 1), dec_hidden, context_vector, tf.stack(attentions, 1), tf.stack(p_gens, 1)

        else:
            for t in range(dec_inp.shape[1]):
                dec_x, pred, dec_hidden = self.decoder(tf.expand_dims(dec_inp[:, t], 1),
                                                       dec_hidden,
                                                       enc_output,
                                                       context_vector)
                context_vector, attn = self.attention(dec_hidden, enc_output)
                predictions.append(pred)

            return tf.stack(predictions, 1), dec_hidden


def _calc_final_dist(_enc_batch_extend_vocab, vocab_dists, attn_dists, p_gens, batch_oov_len, vocab_size, batch_size):
    """
    Calculate the final distribution, for the pointer-generator model
    Args:
    vocab_dists: The vocabulary distributions. List length max_dec_steps of (batch_size, vsize) arrays.
                The words are in the order they appear in the vocabulary file.
    attn_dists: The attention distributions. List length max_dec_steps of (batch_size, attn_len) arrays
    Returns:
    final_dists: The final distributions. List length max_dec_steps of (batch_size, extended_vsize) arrays.
    """
    # Multiply vocab dists by p_gen and attention dists by (1-p_gen)
    vocab_dists = [p_gen * dist for (p_gen, dist) in zip(p_gens, vocab_dists)]
    attn_dists = [(1-p_gen) * dist for (p_gen, dist) in zip(p_gens, attn_dists)]

    # Concatenate some zeros to each vocabulary dist, to hold the probabilities for in-article OOV words
    extended_vsize = vocab_size + batch_oov_len # the maximum (over the batch) size of the extended vocabulary
    extra_zeros = tf.zeros((batch_size, batch_oov_len))
    # list length max_dec_steps of shape (batch_size, extended_vsize)
    vocab_dists_extended = [tf.concat(axis=1, values=[dist, extra_zeros]) for dist in vocab_dists]

    # Project the values in the attention distributions onto the appropriate entries in the final distributions
    # This means that if a_i = 0.1 and the ith encoder word is w, and w has index 500 in the vocabulary,
    # then we add 0.1 onto the 500th entry of the final distribution
    # This is done for each decoder timestep.
    # This is fiddly; we use tf.scatter_nd to do the projection
    batch_nums = tf.range(0, limit=batch_size) # shape (batch_size)
    batch_nums = tf.expand_dims(batch_nums, 1) # shape (batch_size, 1)
    attn_len = tf.shape(_enc_batch_extend_vocab)[1] # number of states we attend over
    batch_nums = tf.tile(batch_nums, [1, attn_len]) # shape (batch_size, attn_len)
    indices = tf.stack((batch_nums, _enc_batch_extend_vocab), axis=2) # shape (batch_size, enc_t, 2)
    shape = [batch_size, extended_vsize]
    # list length max_dec_steps (batch_size, extended_vsize)
    attn_dists_projected = [tf.scatter_nd(indices, copy_dist, shape) for copy_dist in attn_dists]

    # Add the vocab distributions and the copy distributions together to get the final distributions
    # final_dists is a list length max_dec_steps; each entry is a tensor shape (batch_size, extended_vsize) giving
    # the final distribution for that decoder timestep
    # Note that for decoder timesteps and examples corresponding to a [PAD] token, this is junk - ignore.
    final_dists = [vocab_dist + copy_dist for (vocab_dist, copy_dist) in zip(vocab_dists_extended, attn_dists_projected)]

    return final_dists


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


