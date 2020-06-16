import tensorflow as tf
from seq2seq_transformer_pgn_tf2.encoders.self_attention_encoder import EncoderLayer
from seq2seq_transformer_pgn_tf2.decoders.self_attention_decoder import DecoderLayer
from seq2seq_transformer_pgn_tf2.layers.transformer import MultiHeadAttention, Embedding
from seq2seq_transformer_pgn_tf2.layers.position import positional_encoding
from seq2seq_transformer_pgn_tf2.layers.common import point_wise_feed_forward_network
from seq2seq_transformer_pgn_tf2.utils.decoding import calc_final_dist
from seq2seq_transformer_pgn_tf2.layers.transformer import create_look_ahead_mask, create_padding_mask


class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers        
        
        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) 
                        for _ in range(num_layers)]
    
        self.dropout = tf.keras.layers.Dropout(rate)
            
    def call(self, x, training, mask):        
        x = self.dropout(x, training=training)
        
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)
        
        return x  # (batch_size, input_seq_len, d_model)


class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size, rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.depth = self.d_model // self.num_heads

        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)
        
        self.Wh = tf.keras.layers.Dense(1)
        self.Ws = tf.keras.layers.Dense(1)
        self.Wx = tf.keras.layers.Dense(1)
        self.V = tf.keras.layers.Dense(1)
        
    def call(self, x, enc_output, training, 
            look_ahead_mask, padding_mask):

        attention_weights = {}        
        out = self.dropout(x, training=training)

        for i in range(self.num_layers):
            out, block1, block2 = self.dec_layers[i](out, enc_output, training,
                                                    look_ahead_mask, padding_mask)
        
            attention_weights['decoder_layer{}_block1'.format(i+1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i+1)] = block2
        
        # x.shape == (batch_size, target_seq_len, d_model)
        
        #context vectors
        enc_out_shape = tf.shape(enc_output)
        context = tf.reshape(enc_output,(enc_out_shape[0], enc_out_shape[1], self.num_heads, self.depth) ) # shape : (batch_size, input_seq_len, num_heads, depth)
        context = tf.transpose(context, [0,2,1,3]) # (batch_size, num_heads, input_seq_len, depth)
        context = tf.expand_dims(context, axis=2)  # (batch_size, num_heads, 1, input_seq_len, depth)
        
        attn = tf.expand_dims(block2, axis=-1)  # (batch_size, num_heads, target_seq_len, input_seq_len, 1)
        context = context * attn # (batch_size, num_heads, target_seq_len, input_seq_len, depth)
        context = tf.reduce_sum(context, axis=3) # (batch_size, num_heads, target_seq_len, depth)
        context = tf.transpose(context, [0,2,1,3]) # (batch_size, target_seq_len, num_heads, depth)
        context = tf.reshape(context, (tf.shape(context)[0], tf.shape(context)[1], self.d_model)) # (batch_size, target_seq_len, d_model)
        
        # P_gens computing
        a = self.Wx(x)
        b = self.Ws(out)
        c = self.Wh(context)
        p_gens = tf.sigmoid(self.V(a + b + c))
        # print('out is ', out)
        # print('attention_weights is ', attention_weights)
        # print('p_gens is ', p_gens)
        return out, attention_weights, p_gens


class PGN_TRANSFORMER(tf.keras.Model):
    def __init__(self, params):
    # def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, 
    #             target_vocab_size, pe_input, pe_target, rate=0.1):
        super(PGN_TRANSFORMER, self).__init__()

        self.params = params
        
        self.embedding = Embedding(params["vocab_size"],
                                   params["d_model"])
        self.encoder = Encoder(
            params["num_blocks"],
            params["d_model"],
            params["num_heads"],
            params["dff"],
            params["vocab_size"],
            params["dropout_rate"])

        self.decoder = Decoder(
            params["num_blocks"],
            params["d_model"],
            params["num_heads"],
            params["dff"], 
            params["vocab_size"],
            params["dropout_rate"])

        self.final_layer = tf.keras.layers.Dense(params["vocab_size"])
    
    def call(self, inp, extended_inp, max_oov_len, tar, training, enc_padding_mask, look_ahead_mask, dec_padding_mask):
        # print('inp is ', inp)
        embed_x = self.embedding(inp)
        embed_dec = self.embedding(tar)
        
        enc_output = self.encoder(embed_x, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)
        
        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights, p_gens = self.decoder(embed_dec,
                                                     enc_output,
                                                     training,
                                                     look_ahead_mask,
                                                     dec_padding_mask)
        # print('dec_output is ', dec_output)
        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)
        final_output = tf.nn.softmax(final_output)
        # print('final_output is ', final_output)
        # p_gens = tf.keras.layers.Dense(tf.concat([before_dec, dec, attn_dists[-1]], axis=-1),units=1,activation=tf.sigmoid,trainable=training,use_bias=False)
        attn_dists = attention_weights['decoder_layer{}_block2'.format(self.params["num_blocks"])] # (batch_size,num_heads, targ_seq_len, inp_seq_len)
        attn_dists = tf.reduce_sum(attn_dists, axis=1)/self.params["num_heads"] # (batch_size, targ_seq_len, inp_seq_len)
        # print('attn_dists is ', attn_dists)
        final_dists = calc_final_dist(extended_inp,
                                      tf.unstack(final_output, axis=1),
                                      tf.unstack(attn_dists, axis=1),
                                      tf.unstack(p_gens, axis=1),
                                      max_oov_len,
                                      self.params["vocab_size"],
                                      self.params["batch_size"])
                                      
        outputs = dict(logits=tf.stack(final_dists, 1), attentions=attn_dists)
        return outputs

