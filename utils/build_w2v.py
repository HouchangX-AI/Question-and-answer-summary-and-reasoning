from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from gensim.models.keyedvectors import KeyedVectors
from seq2seq_tf2 import config
from utils.data_utils import dump_pkl
from seq2seq_tf2.data_reader import read_data, build_vocab, save_word_dict


def get_sentence(sentence_tag, word_sep=' ', pos_sep='|'):
    """
    文本拼接
    :param sentence_tag:
    :param word_sep:
    :param pos_sep:
    :return:
    """
    words = []
    for item in sentence_tag.split(word_sep):
        if pos_sep in item:
            index = item.rindex(pos_sep)
            words.append(item[:index])
        else:
            words.append(item.strip())
    return word_sep.join(words)


def read_lines(path, col_sep=None):
    lines = []
    with open(path, mode='r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if col_sep:
                if col_sep in line:
                    lines.append(line)
            else:
                lines.append(line)
    return lines


def extract_sentence(train_x_seg_path, train_y_seg_path, test_seg_path, col_sep='\t'):
    ret = []
    lines = read_lines(train_x_seg_path)
    lines += read_lines(train_y_seg_path)
    lines += read_lines(test_seg_path)
    for line in lines:
        ret.append(line)
        # if col_sep in line:
        #     index = line.index(col_sep)
        #     word_tag = line[index + 1:]
        #     sentence = ''.join(get_sentence(word_tag))
        #     ret.append(sentence)
    return ret


def save_sentence(lines, sentence_path):
    with open(sentence_path, 'w', encoding='utf-8') as f:
        for line in lines:
            f.write('%s\n' % line.strip())
    print('save sentence:%s' % sentence_path)


def build(train_x_seg_path, test_y_seg_path, test_seg_path, out_path=None, sentence_path='',
          w2v_bin_path="w2v.bin", min_count=1, col_sep='\t'):
    sentences = extract_sentence(train_x_seg_path, test_y_seg_path, test_seg_path, col_sep=col_sep)
    save_sentence(sentences, sentence_path)
    print('train w2v model...')
    # train model
    w2v = Word2Vec(sg=1, sentences=LineSentence(sentence_path),
                   size=256, window=5, min_count=min_count, iter=40)
    w2v.wv.save_word2vec_format(w2v_bin_path, binary=True)
    print("save %s ok." % w2v_bin_path)
    # test
    # sim = w2v.wv.similarity('大', '小')
    # print('大 vs 小 similarity score:', sim)
    # load model
    model = KeyedVectors.load_word2vec_format(w2v_bin_path, binary=True)
    word_dict = {}
    for word in model.vocab:
        word_dict[word] = model[word]
    dump_pkl(word_dict, out_path, overwrite=True)


if __name__ == '__main__':
    build(config.train_seg_path_x,
          config.train_seg_path_y,
          config.test_seg_path_x,
          out_path=config.word2vec_output,
          sentence_path=config.sentence_path,
          w2v_bin_path="w2v.bin",
          min_count=1,
          col_sep='\t')
