import tensorflow as tf
import numpy as np
from seq2seq_tf2.batcher import output_to_words


class Hypothesis:
    """ Class designed to hold hypothesises throughout the beamSearch decoding """

    def __init__(self, tokens, log_probs, state, attn_dists, p_gens, coverage):
        # list of all the tokens from time 0 to the current time step t
        self.tokens = tokens
        # list of the log probabilities of the tokens of the tokens
        self.log_probs = log_probs
        # decoder state after the last token decoding
        self.state = state
        # attention dists of all the tokens
        self.attn_dists = attn_dists
        # generation probability of all the tokens
        self.p_gens = p_gens
        self.coverage = coverage

        # self.abstract = ""
        # self.text = ""
        # self.real_abstract = ""

    def extend(self, token, log_prob, state, attn_dist, p_gen, coverage):
        """Method to extend the current hypothesis by adding the next decoded token and all
        the informations associated with it"""
        return Hypothesis(tokens=self.tokens + [token],  # we add the decoded token
                          log_probs=self.log_probs + [log_prob],  # we add the log prob of the decoded token
                          state=state,  # we update the state
                          attn_dists=self.attn_dists + [attn_dist],
                          # we  add the attention dist of the decoded token
                          p_gens=self.p_gens + [p_gen],  # we add the p_gen
                          coverage=coverage)

    @property
    def latest_token(self):
        return self.tokens[-1]

    @property
    def tot_log_prob(self):
        return sum(self.log_probs)

    @property
    def avg_log_prob(self):
        return self.tot_log_prob / len(self.tokens)


def beam_decode(model, batch, vocab, params):

    def decode_onestep(enc_inp, enc_outputs, dec_input, dec_state, enc_extended_inp,
                       batch_oov_len, enc_pad_mask, use_coverage, prev_coverage):
        """
            Method to decode the output step by step (used for beamSearch decoding)
            Args:
                sess : tf.Session object
                batch : current batch, shape = [beam_size, 1, vocab_size( + max_oov_len if pointer_gen)]
                (for the beam search decoding, batch_size = beam_size)
                enc_outputs : hiddens outputs computed by the encoder LSTM
                dec_state : beam_size-many list of decoder previous state, LSTMStateTuple objects,
                shape = [beam_size, 2, hidden_size]
                dec_input : decoder_input, the previous decoded batch_size-many words, shape = [beam_size, embed_size]
                cov_vec : beam_size-many list of previous coverage vector
            Returns: A dictionary of the results of all the ops computations (see below for more details)
        """
        # print('enc_outputs is ', enc_outputs)
        # print('dec_state is ', dec_state)
        # print('enc_inp is ', enc_inp)
        # print('enc_extended_inp is ', enc_extended_inp)
        # print('dec_input is ', dec_input)
        # print('batch_oov_len is ', batch_oov_len)
        # print('enc_pad_mask is ', enc_pad_mask)
        # print('prev_coverage is ', prev_coverage)
        final_dists, dec_hidden, attentions, coverages, p_gens = model(enc_outputs,  # shape=(3, 115, 256)
                                                                       dec_state,  # shape=(3, 256)
                                                                       enc_inp,  # shape=(3, 115)
                                                                       enc_extended_inp,  # shape=(3, 115)
                                                                       dec_input,  # shape=(3, 1)
                                                                       batch_oov_len,  # shape=()
                                                                       enc_pad_mask,  # shape=(3, 115)
                                                                       use_coverage,
                                                                       prev_coverage)  # shape=(3, 115, 1)
        # final_dists shape=(3, 1, 30000)
        # print('final_dists is ', final_dists)
        # print('dec_hidden is ', dec_hidden)
        # print('attentions is ', attentions)
        # print('coverages is ', coverages)
        # print('p_gens id ', p_gens)
        top_k_probs, top_k_ids = tf.nn.top_k(tf.squeeze(final_dists), k=params["beam_size"] * 2)
        top_k_log_probs = tf.math.log(top_k_probs)
        results = {"dec_state": dec_hidden,
                   "attention_vec": attentions,  # [batch_sz, max_len_x, 1]
                   "top_k_ids": top_k_ids,
                   "top_k_log_probs": top_k_log_probs,
                   "p_gen": p_gens,
                   "prev_coverage": coverages}
        return results

    # end of the nested class

    # We run the encoder once and then we use the results to decode each time step token
    # state shape=(3, 256), enc_outputs shape=(3, 115, 256)
    enc_outputs, state = model.call_encoder(batch[0]["enc_input"])
    # Initial Hypothesises (beam_size many list)
    hyps = [Hypothesis(tokens=[vocab.word_to_id('[START]')],
                       # we initalize all the beam_size hypothesises with the token start
                       log_probs=[0.0],  # Initial log prob = 0
                       # state=state[0],
                       state=state[0],
                       # initial dec_state (we will use only the first dec_state because they're initially the same)
                       attn_dists=[],
                       p_gens=[],  # we init the coverage vector to zero
                       coverage=np.zeros([batch[0]["enc_input"].shape[1], 1], dtype=np.float32),
                       ) for _ in range(params['batch_size'])]  # batch_size == beam_size

    results = []  # list to hold the top beam_size hypothesises
    steps = 0  # initial step

    while steps < params['max_dec_steps'] and len(results) < params['beam_size']:
        # print('step is ', steps)
        latest_tokens = [h.latest_token for h in hyps]  # latest token for each hypothesis , shape : [beam_size]
        # print('latest_tokens is ', latest_tokens)
        # we replace all the oov is by the unknown token
        latest_tokens = [t if t in range(params['vocab_size']) else vocab.word_to_id('[UNK]') for t in latest_tokens]
        # we collect the last states for each hypothesis
        states = [h.state for h in hyps]
        prev_coverage = [h.coverage for h in hyps]  # list of coverage vectors (or None)
        prev_coverage = tf.convert_to_tensor(prev_coverage)

        # we decode the top likely 2 x beam_size tokens tokens at time step t for each hypothesis
        # model, batch, vocab, params
        dec_input = tf.expand_dims(latest_tokens, axis=1)
        # print('dec_input is ', dec_input)
        # print('states is ', states)
        dec_states = tf.stack(states, axis=0)
        # print('dec_states is ', dec_states)
        returns = decode_onestep(batch[0]['enc_input'],  # shape=(3, 115)
                                 enc_outputs,  # shape=(3, 115, 256)
                                 dec_input,  # shape=(3, 1)
                                 dec_states,  # shape=(3, 256)
                                 batch[0]['extended_enc_input'],  # shape=(3, 115)
                                 batch[0]['max_oov_len'],  # shape=()
                                 batch[0]['sample_encoder_pad_mask'],  # shape=(3, 115)
                                 params['is_coverage'],  # true
                                 prev_coverage)  # shape=(3, 115, 1)
        # print('returns["p_gen"] is ', returns["p_gen"])
        # print(np.squeeze(returns["p_gen"]))
        topk_ids, topk_log_probs, new_states, attn_dists, p_gens, prev_coverages = returns['top_k_ids'],\
                                                                                   returns['top_k_log_probs'],\
                                                                                   returns['dec_state'],\
                                                                                   returns['attention_vec'],\
                                                                                   np.squeeze(returns["p_gen"]),\
                                                                                   returns['prev_coverage']
        all_hyps = []
        num_orig_hyps = 1 if steps == 0 else len(hyps)
        num = 1
        for i in range(num_orig_hyps):
            h, new_state, attn_dist, p_gen, coverage = hyps[i], new_states[i], attn_dists[i], p_gens[i], prev_coverages[i]
            # print('h is ', h)
            # print('new_state is ', new_state) shape=(256,)
            # print('attn_dist ids ', attn_dist) shape=(115,)
            # print('p_gen is ', p_gen) 0.4332452
            # print('coverage is ', coverage)shape=(115, 1),
            num += 1
            # print('num is ', num)
            for j in range(params['beam_size'] * 2):
                # we extend each hypothesis with each of the top k tokens
                # (this gives 2 x beam_size new hypothesises for each of the beam_size old hypothesises)
                # print('topk_ids is ', topk_ids) shape=(3, 6)
                # print('topk_log_probs is ', topk_log_probs)shape=(3, 6)
                # print(topk_ids[i, j].numpy())
                # print(topk_log_probs[i, j].numpy())
                new_hyp = h.extend(token=topk_ids[i, j].numpy(),
                                   log_prob=topk_log_probs[i, j].numpy(),
                                   state=new_state,
                                   attn_dist=attn_dist,
                                   p_gen=p_gen,
                                   coverage=coverage)
                all_hyps.append(new_hyp)
        # in the following lines, we sort all the hypothesises, and select only the beam_size most likely hypothesises
        hyps = []
        sorted_hyps = sorted(all_hyps, key=lambda h: h.avg_log_prob, reverse=True)
        for h in sorted_hyps:
            if h.latest_token == vocab.word_to_id('[STOP]'):
                if steps >= params['min_dec_steps']:
                    results.append(h)
            else:
                # print(h.latest_token)
                hyps.append(h)
            if len(hyps) == params['beam_size'] or len(results) == params['beam_size']:
                break

        steps += 1

    if len(results) == 0:
        results = hyps

    # At the end of the loop we return the most likely hypothesis, which holds the most likely ouput sequence,
    # given the input fed to the model
    hyps_sorted = sorted(results, key=lambda h: h.avg_log_prob, reverse=True)
    best_hyp = hyps_sorted[0]
    best_hyp.abstract = " ".join(output_to_words(best_hyp.tokens, vocab, batch[0]["article_oovs"][0])[1:-1])
    best_hyp.text = batch[0]["article"].numpy()[0].decode()
    print('best_hyp is ', best_hyp.abstract)
    return best_hyp
