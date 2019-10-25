from collections import Counter
import io
import tensorflow as tf
from collections import defaultdict
from seq2seq_tf2 import config


# def load_word_dict(save_path):
#     dict_data = dict()
#     with open(save_path, 'r', encoding='utf-8') as f:
#         for line in f:
#             items = line.strip().split()
#             try:
#                 dict_data[items[0]] = int(items[1])
#             except IndexError:
#                 print('error', line)
#     return dict_data
#
#
def save_word_dict(vocab, save_path):
    with open(save_path, 'w', encoding='utf-8') as f:
        for line in vocab:
            w, i = line
            f.write("%s\t%d\n" % (w, i))
#
#
# def read_samples_by_string(path):
#     with open(path, 'r', encoding='utf-8') as f:
#         for line in f:
#             line = line.strip()
#             if not line:
#                 continue
#             parts = line.lower().strip().split('\t')
#             if len(parts) == 3:
#                 question, dialogue, report = parts[0], parts[1], parts[2]
#                 yield question, dialogue, report
#             if len(parts) == 2:
#                 question, dialogue = parts[0], parts[1]
#                 yield question, dialogue
#
#
# def build_dataset(path):
#     print('Read data, path:{0}'.format(path))
#     question, dialogue, report = [], [], []
#     for q, d, r in read_samples_by_string(path):
#         question.append(q)
#         dialogue.append(d)
#         report.append(r)
#     return question, dialogue, report
#
#
# def build_test_dataset(path):
#     print('Read data, path:{0}'.format(path))
#     question, dialogue = [], []
#     for q, d in read_samples_by_string(path):
#         question.append(q)
#         dialogue.append(d)
#     return question, dialogue
#
#
# def create_dataset(path, num_examples):
#     lines = io.open(path, encoding='UTF-8').read().strip().split('\n')
#     word_pairs = [[preprocess_sentence(w) for w in l.split('\t')[:-1]] for l in lines[:num_examples]]
#     question, dialogue, report = [], [], []
#     for i in word_pairs:
#         question.append(i[0])
#         dialogue.append(i[1])
#         if len(i) == 3:
#             report.append(i[2])
#         else:
#             report.append('<start> ' + 'none' + ' <end>')
#
#     # report = [i[2] for i in word_pairs]
#     return question, dialogue, report
#
#
# def load_dataset(path, num_examples=None):
#     # creating cleaned input, output pairs
#     # question, dialogue, report = build_dataset(path)
#     # inp_lang = question + dialogue
#     # targ_lang = report
#     # targ_lang, inp_lang = create_dataset(path, num_examples)
#     question, dialogue, report = create_dataset(path, num_examples)
#     # print(question, dialogue, report)
#     # targ_lang, inp_lang = create_dataset(path, num_examples)
#     input_tensor, inp_lang_tokenizer = tokenize(dialogue)
#     target_tensor, targ_lang_tokenizer = tokenize(report)
#
#     return input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer
#
#
# def tokenize(lang):
#     lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
#     lang_tokenizer.fit_on_texts(lang)
#     tensor = lang_tokenizer.texts_to_sequences(lang)
#     tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')
#     return tensor, lang_tokenizer


def load_data(path):
    src, tgt = [], []
    with open(path, encoding='utf-8') as f:
        for line in f:
            src_, tgt_ = line.strip().split('\t')
            src.append(preprocess_text(src_))
            tgt.append(preprocess_text(tgt_))
    return src, tgt


def read_data(path_1, path_2, path_3):
    with open(path_1, 'r', encoding='utf-8') as f1, \
            open(path_2, 'r', encoding='utf-8') as f2, \
            open(path_3, 'r', encoding='utf-8') as f3:
        words = []
        # print(f1)
        for line in f1:
            words = line.split()

        for line in f2:
            words += line.split(' ')

        for line in f3:
            words += line.split(' ')

    return words


def build_vocab(items, sort=True, min_count=0, lower=False):
    """
    构建词典列表
    :param items: list  [item1, item2, ... ]
    :param sort: 是否按频率排序，否则按items排序
    :param min_count: 词典最小频次
    :param lower: 是否小写
    :return: list: word set
    """
    result = []
    if sort:
        # sort by count
        dic = defaultdict(int)
        for item in items:
            for i in item.split(" "):
                i = i.strip()
                if not i: continue
                i = i if not lower else item.lower()
                dic[i] += 1
        # sort
        dic = sorted(dic.items(), key=lambda d: d[1], reverse=True)
        for i, item in enumerate(dic):
            key = item[0]
            if min_count and min_count > item[1]:
                continue
            result.append(key)
    else:
        # sort by items
        for i, item in enumerate(items):
            item = item if not lower else item.lower()
            result.append(item)

    vocab = [(w, i) for i, w in enumerate(result)]
    reverse_vocab = [(i, w) for i, w in enumerate(result)]

    return vocab, reverse_vocab


def load_data(path, col_sep='\t', word_sep=' ', pos_sep='/'):
    lines = read_lines(path, col_sep)
    word_lst = []
    pos_lst = []
    label_lst = []
    for line in lines:
        index = line.index(col_sep)
        label = line[:index]
        if pos_sep in label:
            label = label.split(pos_sep)[0]
        label_lst.append(label)
        sentence = line[index + 1:]
        # word and pos
        word_pos_list = sentence.split(word_sep)
        word, pos = [], []
        for item in word_pos_list:
            if pos_sep in item:
                r_index = item.rindex(pos_sep)
                w, p = item[:r_index], item[r_index + 1:]
                if w == '' or p == '':
                    continue
                word.append(w)
                pos.append(p)
            else:
                word.append(item.strip())
        word_lst.extend(word)
        pos_lst.extend(pos)
    return word_lst, pos_lst, label_lst


if __name__ == '__main__':
    lines = read_data(config.train_seg_path_x, config.train_seg_path_y, config.test_seg_path_x)
    vocab, reverse_vocab = build_vocab(lines)
    save_word_dict(vocab, config.vocab_path)