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


def parse_data(path, key_location=3):

    pd_data = pd.read_csv(path, encoding='utf-8')
    np_data = np.array(pd_data)
    data_list = np_data.tolist()[:100]
    # print(data_list)

    results_1 = []
    results_2 = []
    for i in data_list:
        if len(i) == 6:
            results_1.append(i[3] + " " + i[4])
            results_2.append(i[5])
    print(results_1)
    print(results_2)
    return results_1, results_2


def save_data(data_1, data_2, data_path_1, data_path_2, stop_words_path=''):
    stopwords = read_stopwords(stop_words_path)
    with open(data_path_1, 'w', encoding='utf-8') as f1, open(data_path_2, 'w', encoding='utf-8') as f2:
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


if __name__ == '__main__':
    data_list_1, data_list_2 = parse_data(config.train_path)
    save_data(data_list_1, data_list_2, config.train_seg_path_x, config.train_seg_path_y, stop_words_path=config.stop_words_path)
