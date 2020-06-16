import tensorflow as tf
import numpy as np
from seq2seq_pgn_tf2.batcher import output_to_words
from tqdm import tqdm
import math


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
        outputs = model(enc_outputs,  # shape=(3, 115, 256)
                        dec_state,  # shape=(3, 256)
                        enc_inp,  # shape=(3, 115)
                        enc_extended_inp,  # shape=(3, 115)
                        dec_input,  # shape=(3, 1)
                        batch_oov_len,  # shape=()
                        enc_pad_mask,  # shape=(3, 115)
                        use_coverage,
                        prev_coverage)  # shape=(3, 115, 1)
        final_dists=outputs["logits"]
        dec_hidden=outputs["dec_hidden"]
        attentions=outputs["attentions"]
        coverages=outputs["coverages"]
        p_gens=outputs["p_gens"]

        # final_dists shape=(3, 1, 30000)
        # top_k_probs shape=(3, 6)
        # top_k_ids shape=(3, 6)
        top_k_probs, top_k_ids = tf.nn.top_k(tf.squeeze(final_dists), k=params["beam_size"] * 2)
        top_k_log_probs = tf.math.log(top_k_probs)
        # dec_hidden shape = (3, 256)
        # attentions, shape = (3, 115)
        # p_gens shape = (3, 1)
        # coverages,shape = (3, 115, 1)
        results = {"dec_state": dec_hidden,
                   "attention_vec": attentions,  # [batch_sz, max_len_x, 1]
                   "top_k_ids": top_k_ids,
                   "top_k_log_probs": top_k_log_probs,
                   "p_gen": p_gens,
                   "coverages": coverages
                   }
        return results

    # end of the nested class

    # We run the encoder once and then we use the results to decode each time step token
    # state shape=(3, 256), enc_outputs shape=(3, 115, 256)
    enc_input = batch[0]["enc_input"]
    enc_outputs, state = model.call_encoder(enc_input)
    # Initial Hypothesises (beam_size many list)
    hyps = [Hypothesis(tokens=[vocab.word_to_id('[START]')],
                       log_probs=[0.0],
                       state=state[0],
                       p_gens=[],
                       attn_dists=[],
                       coverage=np.zeros([enc_input.shape[1], 1], dtype=np.float32)) for _ in range(params['batch_size'])]
    results = []  # list to hold the top beam_size hypothesises
    steps = 0  # initial step
    while steps < params['max_dec_steps'] and len(results) < params['beam_size']:
        latest_tokens = [h.latest_token for h in hyps]  # latest token for each hypothesis , shape : [beam_size]
        # we replace all the oov is by the unknown token
        latest_tokens = [t if t in range(params['vocab_size']) else vocab.word_to_id('[UNK]') for t in latest_tokens]
        # we collect the last states for each hypothesis
        states = [h.state for h in hyps]
        # prev_coverage = [h.coverage for h in hyps]  # list of coverage vectors (or None)
        # prev_coverage = tf.convert_to_tensor(prev_coverage)

        # we decode the top likely 2 x beam_size tokens tokens at time step t for each hypothesis
        # model, batch, vocab, params
        dec_input = tf.expand_dims(latest_tokens, axis=1)  # shape=(3, 1)
        dec_states = tf.stack(states, axis=0)
        returns = decode_onestep(batch[0]['enc_input'],  # shape=(3, 115)
                                 enc_outputs,  # shape=(3, 115, 256)
                                 dec_input,  # shape=(3, 1)
                                 dec_states,  # shape=(3, 256)
                                 batch[0]['extended_enc_input'],  # shape=(3, 115)
                                 batch[0]['max_oov_len'],  # shape=()
                                 batch[0]['sample_encoder_pad_mask'],  # shape=(3, 115)
                                 params['is_coverage'],  # true
                                 prev_coverage=None)  # shape=(3, 115, 1)
        topk_ids, topk_log_probs, new_states, attn_dists, p_gens, new_coverage= returns['top_k_ids'],\
                                                                    returns['top_k_log_probs'],\
                                                                                   returns['dec_state'],\
                                                                                   returns['attention_vec'],\
                                                                                   returns["p_gen"],\
                                                                                       returns['coverages']                                                                               
        all_hyps = []
        num_orig_hyps = 1 if steps == 0 else len(hyps)
        num = 1
        for i in range(num_orig_hyps):
            h = hyps[i]
            new_state = new_states[i]
            attn_dist = attn_dists[i]
            p_gen = p_gens[i]
            new_coverage_i = new_coverage[i]
            num += 1
            for j in range(params['beam_size'] * 2):
                # we extend each hypothesis with each of the top k tokens
                # (this gives 2 x beam_size new hypothesises for each of the beam_size old hypothesises)
                new_hyp = h.extend(token=topk_ids[i, j].numpy(),
                                   log_prob=topk_log_probs[i, j],
                                   state=new_state,
                                   attn_dist=attn_dist,
                                   p_gen=p_gen,
                                   coverage=new_coverage_i
                                   )
                all_hyps.append(new_hyp)
        # in the following lines, we sort all the hypothesises, and select only the beam_size most likely hypothesises
        hyps = []
        sorted_hyps = sorted(all_hyps, key=lambda h: h.avg_log_prob, reverse=True)
        for h in sorted_hyps:
            if h.latest_token == vocab.word_to_id('[STOP]'):
                if steps >= params['min_dec_steps']:
                    results.append(h)
            else:
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
    best_hyp = result_index2text(best_hyp, vocab, batch)
    # best_hyp.abstract = " ".join(output_to_words(best_hyp.tokens, vocab, batch[0]["article_oovs"][0])[1:-1])
    # best_hyp.text = batch[0]["article"].numpy()[0].decode()
    return best_hyp


def result_index2text(hyp, vocab, batch):
    article_oovs = batch[0]["article_oovs"].numpy()[0]
    hyp.real_abstract = batch[1]["abstract"].numpy()[0].decode()
    hyp.article = batch[0]["article"].numpy()[0].decode()

    words = []
    for index in hyp.tokens:
        if index != 2 and index != 3:
            if index < (len(article_oovs) + vocab.size()):
                if index < vocab.size():
                    words.append(vocab.id_to_word(index))
                else:
                    words.append(article_oovs[index - vocab.size()].decode())
            else:
                print('error values id :{}'.format(index))
    hyp.abstract = " ".join(words)
    return hyp