import tensorflow as tf
from seq2seq_pgn_tf2.layers.transformer import MultiHeadAttention
from seq2seq_pgn_tf2.layers.position import positional_encoding
from seq2seq_pgn_tf2.layers.common import point_wise_feed_forward_network
from seq2seq_pgn_tf2.utils.decoding import calc_final_dist


class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                maximum_position_encoding, rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        
        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, 
                                                self.d_model)
        
        
        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) 
                        for _ in range(num_layers)]
    
        self.dropout = tf.keras.layers.Dropout(rate)
            
    def call(self, x, training, mask):

        seq_len = tf.shape(x)[1]
        
        # 将嵌入和位置编码相加。
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)
        
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)
        
        return x  # (batch_size, input_seq_len, d_model)


class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size,
                maximum_position_encoding, rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        
        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)
        
        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate) 
                        for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)
        
    def call(self, x, enc_output, training, 
            look_ahead_mask, padding_mask):

        seq_len = tf.shape(x)[1]
        attention_weights = {}
        
        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        
        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training,
                                                    look_ahead_mask, padding_mask)
        
        attention_weights['decoder_layer{}_block1'.format(i+1)] = block1
        attention_weights['decoder_layer{}_block2'.format(i+1)] = block2
        
        # x.shape == (batch_size, target_seq_len, d_model)
        return x, attention_weights


class PGN_TRANSFORMER(tf.keras.Model):
    def __init__(self, params):
    # def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, 
    #             target_vocab_size, pe_input, pe_target, rate=0.1):
        super(PGN_TRANSFORMER, self).__init__()

        self.params = params
        self.encoder = Encoder(
            params["num_blocks"],
            params["d_model"],
            params["num_heads"],
            params["dff"],
            params["vocab_size"],
            params["max_enc_len"],
            params["dropout_rate"])

        self.decoder = Decoder(
            params["num_blocks"],
            params["d_model"],
            params["num_heads"],
            params["dff"], 
            params["vocab_size"],
            params["max_dec_len"],
            params["dropout_rate"])

        self.final_layer = tf.keras.layers.Dense(params["vocab_size"])
        
    def call(self, inp, tar, training, enc_padding_mask, 
            look_ahead_mask, dec_padding_mask):

        enc_output = self.encoder(inp, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)
        
        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(tar, enc_output, training, look_ahead_mask, dec_padding_mask)
        
        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)
        if self.params["pointer_gen"]:
            final_dists = calc_final_dist(enc_extended_inp,
                                          predictions,
                                          attentions,
                                          p_gens,
                                          batch_oov_len,
                                          self.params["vocab_size"],
                                          self.params["batch_size"])
        outputs = dict(logits=tf.stack(final_dists, 1), attentions=attention_weights)
        return outputs


class PGN_TRANSFORMER(tf.keras.Model):
    def call(self, enc_output, dec_hidden, enc_inp,
             enc_extended_inp, dec_inp, batch_oov_len,
             enc_padding_mask, use_coverage, prev_coverage):
        predictions = []
        attentions = []
        coverages = []
        p_gens = []

        context_vector, attn_dist, coverage_next = self.attention(dec_hidden,  # shape=(16, 256)
                                                                  enc_output,  # shape=(16, 200, 256)
                                                                  enc_padding_mask,  # (16, 200)
                                                                  use_coverage,
                                                                  prev_coverage)  # None
        elif self.params["mode"] == "train":
            for t in range(dec_inp.shape[1]):
                dec_x, pred, dec_hidden = self.decoder(tf.expand_dims(dec_inp[:, t], 1),
                                                       dec_hidden,
                                                       enc_output,
                                                       context_vector)
                context_vector, attn_dist, coverage_next = self.attention(dec_hidden,
                                                                          enc_output,
                                                                          enc_padding_mask,
                                                                          use_coverage,
                                                                          coverage_next)
                predictions.append(pred)
                coverages.append(coverage_next)
                attentions.append(attn_dist)
                if self.params["pointer_gen"]:
                    p_gen = self.pointer(context_vector, dec_hidden, tf.squeeze(dec_x, axis=1))
                    p_gens.append(p_gen)
            if p_gens:
                final_dists = _calc_final_dist(enc_extended_inp, predictions, attentions, p_gens, batch_oov_len,
                                               self.params["vocab_size"], self.params["batch_size"])
                return tf.stack(final_dists, 1), dec_hidden, attentions, coverages
            else:
                return tf.stack(predictions, 1), dec_hidden, attentions, coverages