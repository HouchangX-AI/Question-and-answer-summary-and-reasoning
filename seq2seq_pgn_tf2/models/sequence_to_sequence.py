import tensorflow as tf
from seq2seq_pgn_tf2.encoders import rnn_encoder
from seq2seq_pgn_tf2.decoders import rnn_decoder
from utils.data_utils import load_word2vec


class SequenceToSequence(tf.keras.Model):
    def __init__(self, params):
        super(SequenceToSequence, self).__init__()
        self.embedding_matrix = load_word2vec(params)
        self.params = params
        self.encoder = rnn_encoder.Encoder(params["vocab_size"],
                                           params["embed_size"],
                                           params["enc_units"],
                                           params["batch_size"],
                                           self.embedding_matrix)
        self.attention = rnn_decoder.BahdanauAttention(params["attn_units"])
        self.decoder = rnn_decoder.Decoder(params["vocab_size"],
                               params["embed_size"],
                               params["dec_units"],
                               params["batch_size"],
                               self.embedding_matrix)

    def call_encoder(self, enc_inp):
        enc_hidden = self.encoder.initialize_hidden_state()
        # [batch_sz, max_train_x, enc_units], [batch_sz, enc_units]
        enc_output, enc_hidden = self.encoder(enc_inp, enc_hidden)
        return enc_output, enc_hidden

    def call(self, enc_output, dec_hidden, enc_inp, dec_inp):
        if self.params["mode"] == "train":
            outputs = self._decode_target(enc_output, dec_hidden, dec_inp)
            return outputs
    
    def _decode_target(self, enc_output, dec_hidden, dec_inp):
        predictions = []
        attentions = []
        context_vector, attn_dist = self.attention(dec_hidden,  # shape=(16, 256)
                                                   enc_output) # shape=(16, 200, 256)
        for t in range(dec_inp.shape[1]):
            # Teachering Forcing
            dec_x, pred, dec_hidden = self.decoder(tf.expand_dims(dec_inp[:, t], 1),
                                                   dec_hidden,
                                                   enc_output,
                                                   context_vector)
            context_vector, attn_dist = self.attention(dec_hidden, enc_output)
            predictions.append(pred)
            attentions.append(attn_dist)
        outputs = dict(logits=tf.stack(predictions, 1), dec_hidden=dec_hidden, attentions=attentions)
        return outputs
    
    # def _dynamic_decode(self,
    #                    enc_output,
    #                    dec_hidden,
    #                    enc_inp,
    #                    enc_extended_inp,
    #                    dec_inp,
    #                    batch_oov_len,
    #                    enc_padding_mask,
    #                    use_coverage,
    #                    prev_coverage):
    #     context_vector, attn_dist, coverage_next = self.attention(dec_hidden,  # shape=(16, 256)
    #                                                               enc_output,  # shape=(16, 200, 256)
    #                                                               enc_padding_mask,  # (16, 200)
    #                                                               use_coverage,
    #                                                               prev_coverage)  # None
    #     dec_x, pred, dec_hidden = self.decoder(dec_inp,
    #                                            dec_hidden,
    #                                            enc_output,
    #                                            context_vector)
    #     if self.params["pointer_gen"]:
    #         p_gen = self.pointer(context_vector, dec_hidden, tf.squeeze(dec_x, axis=1))
    #         final_dists = _calc_final_dist(enc_extended_inp,
    #                                            [pred],
    #                                            [attn_dist],
    #                                            [p_gen],
    #                                            batch_oov_len,
    #                                            self.params["vocab_size"],
    #                                            self.params["batch_size"])
    #         outputs = dict(logits=tf.stack(final_dists, 1), dec_hidden=dec_hidden, attentions=attentions, p_gen=p_gen)
    #         return outputs

