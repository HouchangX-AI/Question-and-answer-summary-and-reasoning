import paddle.fluid as fluid
import paddle
import os
import six
from seq2seq_paddle.seq2seq_model import encoder, train_decoder
from utils.data_utils import load_vocab, read_vocab, save_word_dict, build_vocab, write_vocab, transform_data
from seq2seq_paddle.reader import build_dataset, read_data
from seq2seq_paddle import config
from preprocess import seg_data

batch_size = 32
# def data_generator(input_texts, target_texts, vocab2id, batch_size, maxlen=400):
#     # 数据生成器
#     while True:
#         X, Y = [], []
#         for i in range(len(input_texts)):
#             X.append(str2id(input_texts[i], vocab2id, maxlen))
#             Y.append([vocab2id[GO_TOKEN]] + str2id(target_texts[i], vocab2id, maxlen) + [vocab2id[EOS_TOKEN]])
#             if len(X) == batch_size:
#                 X = np.array(padding(X, vocab2id))
#                 Y = np.array(padding(Y, vocab2id))
#                 yield [X, Y], None
#                 X, Y = [], []


def data_generator(data):
    # 数据生成器
    def reader():
        for q_ids, d_ids, r_ids in data:
            yield q_ids, d_ids, r_ids[:-1], r_ids[1:]
    return reader


def train_model():
    encoded_question_vector, encoded_dialogue_vector = encoder()
    rnn_out = train_decoder(encoded_question_vector, encoded_dialogue_vector)
    label = fluid.layers.data(name="report_next_word", shape=[1], dtype='int64', lod_level=1)
    cost = fluid.layers.cross_entropy(input=rnn_out, label=label)
    avg_cost = fluid.layers.mean(cost)
    return avg_cost


def optimizer_func(hidden_dim):
    fluid.clip.set_gradient_clip(clip=fluid.clip.GradientClipByGlobalNorm(clip_norm=5.0))
    lr_decay = fluid.layers.learning_rate_scheduler.noam_decay(hidden_dim, 1000)
    return fluid.optimizer.Adam(learning_rate=lr_decay,
                                regularization=fluid.regularizer.L2DecayRegularizer(regularization_coeff=1e-4))


def train(save_vocab_path='',
          train_path='',
          test_path='',
          train_seg_path='',
          test_seg_path='',
          model_save_dir='',
          vocab_max_size=5000,
          vocab_min_count=5,
          hidden_dim=512,
          use_cuda=False):

    train_prog = fluid.Program()
    startup_prog = fluid.Program()
    with fluid.program_guard(train_prog, startup_prog):
        with fluid.unique_name.guard():
            avg_cost = train_model()
            optimizer = optimizer_func(hidden_dim)
            optimizer.minimize(avg_cost)

    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    exe = fluid.Executor(place)

    seg_data(train_path, test_path)
    train_texts = build_dataset(train_seg_path)

    if os.path.exists(save_vocab_path):
        vocab = load_vocab(save_vocab_path)
    else:
        vocab, reverse_vocab = build_vocab(train_texts, min_count=vocab_min_count)
        write_vocab(vocab, save_vocab_path)
        vocab = load_vocab(save_vocab_path)

    train_set = read_data(train_seg_path)
    train_set_ids = transform_data(train_set, vocab)
    num_encoder_tokens = len(train_set_ids)
    max_input_texts_len = max([len(text) for text in train_texts])
    print('num of samples:', len(train_texts))
    print('num of unique input tokens:', num_encoder_tokens)
    print('max sequence length for inputs:', max_input_texts_len)
    # save_word_dict(vocab2id, save_vocab_path)

    train_reader = data_generator(train_set_ids)

    train_data = paddle.batch(paddle.reader.shuffle(train_reader, buf_size=10000), batch_size=batch_size)

    feeder = fluid.DataFeeder(feed_list=['question_word', 'dialogue_word', 'report_word', 'report_next_word'],
                              place=place,
                              program=train_prog)

    exe.run(startup_prog)

    EPOCH_NUM = 20
    for pass_id in six.moves.xrange(EPOCH_NUM):
        batch_id = 0
        for data in train_data():
            cost = exe.run(train_prog, feed=feeder.feed(data), fetch_list=[avg_cost])[0]
            print('pass_id: %d, batch_id: %d, loss: %f' % (pass_id, batch_id, cost))
            batch_id += 1
        fluid.io.save_params(exe, model_save_dir, main_program=train_prog)

 
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


if __name__ == '__main__':
    train(save_vocab_path=config.save_vocab_path,
          train_path=config.train_path,
          test_path=config.test_path,
          train_seg_path=config.train_seg_path,
          test_seg_path=config.test_seg_path,
          model_save_dir=config.model_save_dir,
          vocab_max_size=config.vocab_max_size,
          vocab_min_count=config.vocab_min_count,
          hidden_dim=config.hidden_dim,
          use_cuda=config.use_cuda)
