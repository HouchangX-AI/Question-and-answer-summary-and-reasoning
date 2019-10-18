import paddle.fluid as fluid
import paddle
import jieba
import pandas as pd
import copy
import codecs
from gensim.models import Word2Vec
import json
import numpy as np


start_token = u"<s>"
end_token = u"<e>"
unk_token = u"<unk>"

start_id = 0
end_id = 1
unk_id = 2

dict_size = 30000
source_dict_size = target_dict_size = dict_size
word_dim = 512
embedding_size = hidden_dim = 512
decoder_size = hidden_dim
max_question_len = 100
max_dialogue_len = 800
max_report_len = 100
# max_length = 256
beam_size = 4
batch_size = 64
min_count = 5
max_length = 256

vocab_path = "vocab_m%s.txt" %(min_count)

is_sparse = True
word2vec_file="../gensim.sg.%d.txt" %(embedding_size)
data_path = "data_set_m%s.json" %(min_count)
model_save_dir = "machine_translation.inference.model"

embedding_param = fluid.ParamAttr(name='embedding')
dropout_rate = 0.


# def transform_data(data, vocab):
#     # transform sent to ids
#     out_data = []
#     for d in data:
#         tmp_d = []
#         for sent in d:
#             tmp_d.append([vocab.get(t, unk_id) for t in sent if t])
#         out_data.append(tmp_d)
#     return out_data
#
#
# def save_model(trainer, save_path):
#     with open(save_path, 'w') as f:
#         trainer.save_parameter_to_tar(f)
#
#
# def load_word2vec_file(word2vec_file):
#     word2vec_dict = {}
#     input = codecs.open(word2vec_file, "r", "utf-8")
#     lines = input.readlines()
#     input.close()
#     word_num, dim = lines[0].split(" ")
#     word_num = int(word_num)
#     dim = int(dim)
#
#     lines = lines[1:]
#     for l in lines:
#         l = l.strip()
#         tokens = l.split(" ")
#         if len(tokens) != dim + 1:
#             continue
#         w = tokens[0]
#         v = np.array(map(lambda x: float(x), tokens[1:]))
#         word2vec_dict[w] = v
#     return word2vec_dict, dim


def embedding_layer(word_id):
    embed = fluid.layers.embedding(input=word_id,
                                   size=[source_dict_size, word_dim],
                                   dtype='float32',
                                   is_sparse=is_sparse,
                                   param_attr=embedding_param)

    return fluid.layers.dropout(embed, dropout_prob=dropout_rate)


def bigru_layer(embedding):
    fc_forward = fluid.layers.fc(input=embedding, size=hidden_dim * 3, bias_attr=False)
    forward = fluid.layers.dynamic_gru(input=fc_forward,
                                       size=hidden_dim,
                                       param_attr=fluid.ParamAttr(name='gru_forward_encoder'))

    fc_backward = fluid.layers.fc(input=embedding, size=hidden_dim * 3, bias_attr=False)
    backward = fluid.layers.dynamic_gru(input=fc_backward,
                                        size=hidden_dim,
                                        param_attr=fluid.ParamAttr(name='gru_backward_encoder'),
                                        is_reverse=True)

    return forward, backward


def encoder():
    question_word_id = fluid.layers.data(name="question_word", shape=[1], dtype='int64', lod_level=1)
    question_embedding = embedding_layer(question_word_id)
    question_forward, question_backward = bigru_layer(question_embedding)
    encoded_question_vector = fluid.layers.concat(input=[question_forward, question_backward], axis=1)
    encoded_question_vector = fluid.layers.dropout(encoded_question_vector, dropout_prob=dropout_rate)

    dialogue_word_id = fluid.layers.data(name="dialogue_word", shape=[1], dtype='int64', lod_level=1)
    dialogue_embedding = embedding_layer(dialogue_word_id)
    dialogue_forward, dialogue_backward = bigru_layer(dialogue_embedding)
    encoded_dialogue_vector = fluid.layers.concat(input=[dialogue_forward, dialogue_backward], axis=1)
    encoded_dialogue_vector = fluid.layers.dropout(encoded_dialogue_vector, dropout_prob=dropout_rate)

    # encoded_vector = fluid.layers.concat(input=[encoded_question_vector, encoded_dialogue_vector], axis=1)
    return encoded_question_vector, encoded_dialogue_vector


def cell(x, hidden, encoder_out, encoder_out_proj):
    def simple_attention(encoder_vec, encoder_proj, decoder_state):
        decoder_state_proj = fluid.layers.fc(input=decoder_state, size=decoder_size, bias_attr=False)
        decoder_state_expand = fluid.layers.sequence_expand(x=decoder_state_proj, y=encoder_proj)
        mixed_state = fluid.layers.elementwise_add(encoder_proj, decoder_state_expand)
        attention_weights = fluid.layers.fc(input=mixed_state, size=1, bias_attr=False)
        attention_weights = fluid.layers.sequence_softmax(input=attention_weights)
        weigths_reshape = fluid.layers.reshape(x=attention_weights, shape=[-1])
        scaled = fluid.layers.elementwise_mul(x=encoder_vec, y=weigths_reshape, axis=0)
        context = fluid.layers.sequence_pool(input=scaled, pool_type='sum')
        return context

    context = simple_attention(encoder_out, encoder_out_proj, hidden)
    out = fluid.layers.fc(input=[x, context], size=decoder_size * 3, bias_attr=False)
    out = fluid.layers.gru_unit(input=out, hidden=hidden, size=decoder_size * 3)[0]
    return out, out


def train_decoder(encoded_question_vector, encoded_dialogue_vector):
    encoder_last = fluid.layers.sequence_last_step(input=encoded_dialogue_vector)
    encoder_last_proj = fluid.layers.fc(input=encoder_last, size=decoder_size, act='tanh')

    # cache the encoder_out's computed result in attention
    encoder_out_proj = fluid.layers.fc(input=encoded_dialogue_vector, size=decoder_size, bias_attr=False)

    report_word_id = fluid.layers.data(name="report_word", shape=[1], dtype='int64', lod_level=1)
    report_embedding = embedding_layer(report_word_id)

    rnn = fluid.layers.DynamicRNN()
    with rnn.block():
        x = rnn.step_input(report_embedding)
        pre_state = rnn.memory(init=encoder_last_proj, need_reorder=True)
        encoder_out = rnn.static_input(encoded_dialogue_vector)
        encoder_out_proj = rnn.static_input(encoder_out_proj)
        out, current_state = cell(x, pre_state, encoder_out, encoder_out_proj)
        prob = fluid.layers.fc(input=[out], size=target_dict_size, act='softmax')

        rnn.update_memory(pre_state, current_state)
        rnn.output(prob)

    return rnn()


def train_model():
    encoded_question_vector, encoded_dialogue_vector = encoder()
    rnn_out = train_decoder(encoded_question_vector, encoded_dialogue_vector)
    label = fluid.layers.data(name="report_next_word", shape=[1], dtype='int64', lod_level=1)
    cost = fluid.layers.cross_entropy(input=rnn_out, label=label)
    avg_cost = fluid.layers.mean(cost)
    return avg_cost


def optimizer_func():
    fluid.clip.set_gradient_clip(clip=fluid.clip.GradientClipByGlobalNorm(clip_norm=5.0))
    lr_decay = fluid.layers.learning_rate_scheduler.noam_decay(hidden_dim, 1000)
    return fluid.optimizer.Adam(learning_rate=lr_decay,
                                regularization=fluid.regularizer.L2DecayRegularizer(regularization_coeff=1e-4))


def infer_decoder(encoded_question_vector, encoded_dialogue_vector):
    encoder_last = fluid.layers.sequence_last_step(input=encoded_dialogue_vector)
    encoder_last_proj = fluid.layers.fc(input=encoder_last, size=decoder_size, act='tanh')
    encoder_out_proj = fluid.layers.fc(input=encoded_dialogue_vector, size=decoder_size, bias_attr=False)

    max_len = fluid.layers.fill_constant(shape=[1], dtype='int64', value=max_length)
    counter = fluid.layers.zeros(shape=[1], dtype='int64', force_cpu=True)

    init_ids = fluid.layers.data(name="init_ids", shape=[1], dtype="int64", lod_level=2)
    init_scores = fluid.layers.data(name="init_scores", shape=[1], dtype="float32", lod_level=2)
    # create and init arrays to save selected ids, scores and states for each step
    ids_array = fluid.layers.array_write(init_ids, i=counter)
    scores_array = fluid.layers.array_write(init_scores, i=counter)
    state_array = fluid.layers.array_write(encoder_last_proj, i=counter)

    cond = fluid.layers.less_than(x=counter, y=max_len)
    while_op = fluid.layers.While(cond=cond)
    with while_op.block():
        pre_ids = fluid.layers.array_read(array=ids_array, i=counter)
        pre_score = fluid.layers.array_read(array=scores_array, i=counter)
        pre_state = fluid.layers.array_read(array=state_array, i=counter)

        pre_ids_emb = fluid.layers.embedding(input=pre_ids,
                                             size=[target_dict_size, word_dim],
                                             dtype='float32',
                                             is_sparse=is_sparse)
        out, current_state = cell(pre_ids_emb, pre_state, encoded_dialogue_vector, encoder_out_proj)
        prob = fluid.layers.fc(input=current_state, size=target_dict_size, act='softmax')

        # beam search
        topk_scores, topk_indices = fluid.layers.topk(prob, k=beam_size)
        accu_scores = fluid.layers.elementwise_add(x=fluid.layers.log(topk_scores),
                                                   y=fluid.layers.reshape(pre_score, shape=[-1]),
                                                   axis=0)
        accu_scores = fluid.layers.lod_reset(x=accu_scores, y=pre_ids)
        selected_ids, selected_scores = fluid.layers.beam_search(pre_ids, pre_score, topk_indices, accu_scores, beam_size, end_id=1)

        fluid.layers.increment(x=counter, value=1, in_place=True)
        # save selected ids and corresponding scores of each step
        fluid.layers.array_write(selected_ids, array=ids_array, i=counter)
        fluid.layers.array_write(selected_scores, array=scores_array, i=counter)
        # update rnn state by sequence_expand acting as gather
        current_state = fluid.layers.sequence_expand(current_state, selected_ids)
        fluid.layers.array_write(current_state, array=state_array, i=counter)
        current_enc_out = fluid.layers.sequence_expand(encoded_dialogue_vector, selected_ids)
        fluid.layers.assign(current_enc_out, encoded_dialogue_vector)
        current_enc_out_proj = fluid.layers.sequence_expand(encoder_out_proj, selected_ids)
        fluid.layers.assign(current_enc_out_proj, encoder_out_proj)

        # update conditional variable
        length_cond = fluid.layers.less_than(x=counter, y=max_len)
        finish_cond = fluid.layers.logical_not(fluid.layers.is_empty(x=selected_ids))
        fluid.layers.logical_and(x=length_cond, y=finish_cond, out=cond)

    translation_ids, translation_scores = fluid.layers.beam_search_decode(ids=ids_array, scores=scores_array, beam_size=beam_size, end_id=1)

    return translation_ids, translation_scores


def infer_model():
    encoded_question_vector, encoded_dialogue_vector = encoder()
    translation_ids, translation_scores = infer_decoder(encoded_question_vector, encoded_dialogue_vector)
    return translation_ids, translation_scores


def infer(use_cuda):
    infer_prog = fluid.Program()
    startup_prog = fluid.Program()
    with fluid.program_guard(infer_prog, startup_prog):
        with fluid.unique_name.guard():
            translation_ids, translation_scores = infer_model()

    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    exe = fluid.Executor(place)

    test_data = paddle.batch(paddle.dataset.wmt16.test(source_dict_size, target_dict_size),batch_size=batch_size)
    src_idx2word = paddle.dataset.wmt16.get_dict("en", source_dict_size, reverse=True)
    trg_idx2word = paddle.dataset.wmt16.get_dict("de", target_dict_size, reverse=True)

    fluid.io.load_params(exe, model_save_dir, main_program=infer_prog)

    for data in test_data():
        src_word_id = fluid.create_lod_tensor(data=[x[0] for x in data],
                                              recursive_seq_lens=[[len(x[0]) for x in data]],
                                              place=place)
        init_ids = fluid.create_lod_tensor(data=np.array([[0]] * len(data), dtype='int64'),
                                           recursive_seq_lens=[[1] * len(data)] * 2,
                                           place=place)
        init_scores = fluid.create_lod_tensor(data=np.array([[0.]] * len(data), dtype='float32'),
                                              recursive_seq_lens=[[1] * len(data)] * 2,
                                              place=place)
        seq_ids, seq_scores = exe.run(infer_prog,
                                      feed={'src_word_id': src_word_id,'init_ids': init_ids,'init_scores': init_scores},
                                      fetch_list=[translation_ids, translation_scores],
                                      return_numpy=False)
        # How to parse the results:
        #   Suppose the lod of seq_ids is:
        #     [[0, 3, 6], [0, 12, 24, 40, 54, 67, 82]]
        #   then from lod[0]:
        #     there are 2 source sentences, beam width is 3.
        #   from lod[1]:
        #     the first source sentence has 3 hyps; the lengths are 12, 12, 16
        #     the second source sentence has 3 hyps; the lengths are 14, 13, 15
        hyps = [[] for i in range(len(seq_ids.lod()[0]) - 1)]
        scores = [[] for i in range(len(seq_scores.lod()[0]) - 1)]
        for i in range(len(seq_ids.lod()[0]) - 1):  # for each source sentence
            start = seq_ids.lod()[0][i]
            end = seq_ids.lod()[0][i + 1]
            print("Original sentence:")
            print(" ".join([src_idx2word[idx] for idx in data[i][0][1:-1]]))
            print("Translated score and sentence:")
            for j in range(end - start):  # for each candidate
                sub_start = seq_ids.lod()[1][start + j]
                sub_end = seq_ids.lod()[1][start + j + 1]
                hyps[i].append(" ".join([trg_idx2word[idx] for idx in np.array(seq_ids)[sub_start:sub_end][1:-1]]))
                scores[i].append(np.array(seq_scores)[sub_end - 1])
                print(scores[i][-1], hyps[i][-1].encode('utf8'))


def main(use_cuda):
    # train(use_cuda)
    infer(use_cuda)


if __name__ == '__main__':
    use_cuda = False
    main(use_cuda)





