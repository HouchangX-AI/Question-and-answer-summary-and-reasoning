import numpy as np
import pandas as pd
import re
from jieba import posseg
import jieba
from utils.tokenizer import segment


REMOVE_WORDS = ['|', '[', ']', '语音', '图片']


def read_stopwords(path):
    lines = set()
    with open(path, mode='r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            lines.add(line)
    return lines


def remove_words(words_list):
    words_list = [word for word in words_list if word not in REMOVE_WORDS]
    return words_list


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
                seg_list = remove_words(seg_list)
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
                seg_list = remove_words(seg_list)
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
                seg_list = remove_words(seg_list)
                seg_line = ' '.join(seg_list)
                f3.write('%s' % seg_line)
            f3.write('\n')


def preprocess_sentence(sentence):
    seg_list = segment(sentence.strip(), cut_type='word')
    seg_line = ' '.join(seg_list)
    return seg_line


if __name__ == '__main__':
    train_list_src, train_list_trg = parse_data('../datasets/AutoMaster_TrainSet.csv')
    test_list_src, _ = parse_data('../datasets/AutoMaster_TestSet.csv')
    save_data(train_list_src,
              train_list_trg,
              test_list_src,
              '../datasets/train_set.seg_x.txt',
              '../datasets/train_set.seg_y.txt',
              '../datasets/test_set.seg_x.txt',
              stop_words_path='../datasets/stop_words.txt')


