from collections import defaultdict
import numpy as np
import pickle
import os
from collections import Counter
import jieba
# from seq2seq_tf2.data_reader import read_data
from seq2seq_tf2 import config


PAD_TOKEN = 'PAD'
GO_TOKEN = 'GO'
EOS_TOKEN = 'EOS'
UNK_TOKEN = 'UNK'

start_id = 0
end_id = 1
unk_id = 2
# start_token = u"<s>"
# end_token = u"<e>"
# unk_token = u"<unk>"


def save_word_dict(dict_data, save_path):
    with open(save_path, 'w', encoding='utf-8') as f:
        for k, v in dict_data.items():
            f.write("%s\t%d\n" % (k, v))


def read_vocab(input_texts, max_size=50000, min_count=5):
    token_counts = Counter()
    special_tokens = [PAD_TOKEN, GO_TOKEN, EOS_TOKEN, UNK_TOKEN]
    for line in input_texts:
        for char in line.strip():
            char = char.strip()
            if not char:
                continue
            token_counts.update(char)
    # Sort word count by value
    count_pairs = token_counts.most_common()
    vocab = [k for k, v in count_pairs if v >= min_count]
    # Insert the special tokens to the beginning
    vocab[0:0] = special_tokens
    full_token_id = list(zip(vocab, range(len(vocab))))[:max_size]
    vocab2id = dict(full_token_id)
    return vocab2id


def stat_dict(lines):
    word_dict = {}
    for line in lines:
        tokens = line.split(" ")
        for t in tokens:
            t = t.strip()
            if t:
                word_dict[t] = word_dict.get(t, 0) + 1
    return word_dict


def filter_dict(word_dict, min_count=3):
    out_dict = copy.deepcopy(word_dict)
    for w,c in out_dict.items():
        if c < min_count:
            del out_dict[w]
    return out_dict

# def build_vocab(lines, min_count=3):
#     word_dict = stat_dict(lines)
#     word_dict = filter_dict(word_dict, min_count)
#     sorted_dict = sorted(word_dict.items(), key=lambda x:x[1], reverse=True)
#     sorted_words = [w for w,c in sorted_dict]
#     sorted_words = [start_token, end_token, unk_token] + sorted_words
#     vocab = dict([(w,i) for i,w in enumerate(sorted_words)])
#     reverse_vocab = dict([(i,w) for i,w in enumerate(sorted_words)])
#     return vocab, reverse_vocab


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


def load_dict(dict_path):
    return dict((line.strip().split("\t")[0], idx)
                for idx, line in enumerate(open(dict_path, "r", encoding='utf-8').readlines()))


def load_reverse_dict(dict_path):
    return dict((idx, line.strip().split("\t")[0])
                for idx, line in enumerate(open(dict_path, "r", encoding='utf-8').readlines()))


def flatten_list(nest_list):
    """
    嵌套列表压扁成一个列表
    :param nest_list: 嵌套列表
    :return: list
    """
    result = []
    for item in nest_list:
        if isinstance(item, list):
            result.extend(flatten_list(item))
        else:
            result.append(item)
    return result


def map_item2id(items, vocab, max_len, non_word=0, lower=False):
    """
    将word/pos等映射为id
    :param items: list，待映射列表
    :param vocab: 词表
    :param max_len: int，序列最大长度
    :param non_word: 未登录词标号，默认0
    :param lower: bool，小写
    :return: np.array, dtype=int32,shape=[max_len,]
    """
    assert type(non_word) == int
    arr = np.zeros((max_len,), dtype='int32')
    # 截断max_len长度的items
    min_range = min(max_len, len(items))
    for i in range(min_range):
        item = items[i] if not lower else items[i].lower()
        arr[i] = vocab[item] if item in vocab else non_word
    return arr


def write_vocab(vocab, filename):
    """Writes a vocab to a file
    Writes one word per line.
    Args:
        vocab: iterable that yields word
        filename: path to vocab file
    Returns:
        write a word per line
    """
    print("Writing vocab...")
    with open(filename, "w", encoding='utf-8') as f:
        for word, i in sorted(vocab.items(), key=lambda x: x[1]):
            if i != len(vocab) - 1:
                f.write(word + '\n')
            else:
                f.write(word)
    print("- write to {} done. {} tokens".format(filename, len(vocab)))


def load_vocab(filename):
    """Loads vocab from a file
    Args:
        filename: (string) the format of the file must be one word per line.
    Returns:
        d: dict[word] = index
    """
    try:
        d = dict()
        with open(filename, 'r', encoding='utf-8') as f:
            # lines = f.readlines()
            for idx, word in enumerate(f.readlines()):
                word = word.strip()
                d[word] = idx

    except IOError:
        raise IOError(filename)
    return d


def transform_data(data, vocab):
    # transform sent to ids
    out_data = []
    for d in data:
        tmp_d = []
        for sent in d:
            tmp_d.append([vocab.get(t, unk_id) for t in sent if t])
        out_data.append(tmp_d)
    return out_data



def load_pkl(pkl_path):
    """
    加载词典文件
    :param pkl_path:
    :return:
    """
    with open(pkl_path, 'rb') as f:
        result = pickle.load(f)
    return result


def dump_pkl(vocab, pkl_path, overwrite=True):
    """
    存储文件
    :param pkl_path:
    :param overwrite:
    :return:
    """
    if pkl_path and os.path.exists(pkl_path) and not overwrite:
        return
    if pkl_path:
        with open(pkl_path, 'wb') as f:
            pickle.dump(vocab, f, protocol=pickle.HIGHEST_PROTOCOL)
            # pickle.dump(vocab, f, protocol=0)
        print("save %s ok." % pkl_path)


def get_word_segment_data(contents, word_sep=' ', pos_sep='/'):
    data = []
    for content in contents:
        temp = []
        for word in content.split(word_sep):
            if pos_sep in word:
                temp.append(word.split(pos_sep)[0])
            else:
                temp.append(word.strip())
        data.append(word_sep.join(temp))
    return data


def get_char_segment_data(contents, word_sep=' ', pos_sep='/'):
    data = []
    for content in contents:
        temp = ''
        for word in content.split(word_sep):
            if pos_sep in word:
                temp += word.split(pos_sep)[0]
            else:
                temp += word.strip()
        # char seg with list
        data.append(word_sep.join(list(temp)))
    return data


def load_list(path):
    return [word for word in open(path, 'r', encoding='utf-8').read().split()]


def save(pred_labels, ture_labels=None, pred_save_path=None, data_set=None):
    if pred_save_path:
        with open(pred_save_path, 'w', encoding='utf-8') as f:
            for i in range(len(pred_labels)):
                if ture_labels and len(ture_labels) > 0:
                    assert len(ture_labels) == len(pred_labels)
                    if data_set:
                        f.write(ture_labels[i] + '\t' + data_set[i] + '\n')
                    else:
                        f.write(ture_labels[i] + '\n')
                else:
                    if data_set:
                        f.write(pred_labels[i] + '\t' + data_set[i] + '\n')
                    else:
                        f.write(pred_labels[i] + '\n')
        print("pred_save_path:", pred_save_path)


def load_word2vec(vocab_size):
    """
    load pretrain word2vec weight matrix
    :param vocab_size:
    :return:
    """
    word2vec_dict = load_pkl(config.word2vec_output)
    vocab_dict = open(config.vocab_path).readlines()
    embedding_matrix = np.zeros((vocab_size, config.embedding_dim))

    for line in vocab_dict[:vocab_size]:
        word_id = line.split()
        word, i = word_id
        embedding_vector = word2vec_dict.get(word)
        if embedding_vector is not None:
            embedding_matrix[int(i)] = embedding_vector

    return embedding_matrix