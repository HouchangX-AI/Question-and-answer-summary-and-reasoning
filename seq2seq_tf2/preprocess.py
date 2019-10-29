import numpy as np
import pandas as pd
from jieba import posseg
from seq2seq_tf2 import config
import jieba
from tokenizer import segment
from seq2seq_tf2 import config


def read_stopwords(path):
    lines = set()
    with open(path, mode='r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            lines.add(line)
    return lines


def parse_data(path):
    df = pd.read_csv(path, encoding='utf-8')
    data_x = df.Question.str.cat(df.Dialogue)
    data_y = []
    if 'Report' in df.columns:
        data_y = df.Report
    return data_x, data_y


def save_data(data_1, data_2, data_3, data_path_1, data_path_2, data_path_3, stop_words_path=''):
    stopwords = read_stopwords(stop_words_path)
    with open(data_path_1, 'w', encoding='utf-8') as f1:
        count = 0
        for line in data_1:
            # print(line)
            if isinstance(line, str):
                seg_list = segment(line.strip(), cut_type='word')
                # seg_words = []
                # for j in seg_list:
                #     if j in stopwords:
                #         continue
                #     seg_words.append(j)
                seg_line = ' '.join(seg_list)
                f1.write('%s' % seg_line)
            count += 1
            f1.write('\n')

    with open(data_path_2, 'w', encoding='utf-8') as f2:
        for line in data_2:
            if isinstance(line, str):
                seg_list = segment(line.strip(), cut_type='word')
                # seg_words = []
                # for j in seg_list:
                #     if j in stopwords:
                #         continue
                #     seg_words.append(j)
                seg_line = ' '.join(seg_list)
                f2.write('%s' % seg_line)
            f2.write('\n')

    with open(data_path_3, 'w', encoding='utf-8') as f3:
        for line in data_3:
            if isinstance(line, str):
                seg_list = segment(line.strip(), cut_type='word')
                seg_line = ' '.join(seg_list)
                f3.write('%s' % seg_line)
            f3.write('\n')


def preprocess_sentence(sentence):
    seg_list = segment(sentence.strip(), cut_type='word')
    seg_line = ' '.join(seg_list)
    return seg_line


if __name__ == '__main__':
    train_list_src, train_list_trg = parse_data(config.train_path)
    test_list_src, _ = parse_data(config.test_path)
    save_data(train_list_src,
              train_list_trg,
              test_list_src,
              config.train_seg_path_x,
              config.train_seg_path_y,
              config.test_seg_path_x,
              stop_words_path=config.stop_words_path)
