import numpy as np
import pandas as pd
from jieba import posseg
from seq2seq_tf2 import config
import jieba
from tokenizer import segment
from utils.io_utils import get_logger

logger = get_logger(__name__)


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

    results = []
    for i in data_list:
        results.append(i[key_location:])

    return results


def save_data(data, data_path, stop_words_path=''):
    stopwords = read_stopwords(stop_words_path)
    with open(data_path, 'w', encoding='utf-8') as f:

        count = 0
        for line in data:
            if len(line) == 3:
                for i in line:
                    # print(i)
                    if isinstance(i, str):
                        seg_list = segment(i.strip(), cut_type='word')
                        seg_words = []
                        for j in seg_list:
                            if j in stopwords:
                                continue
                            seg_words.append(j)
                        seg_line = ' '.join(seg_words)
                        f.write('%s\t' % seg_line)
                count += 1
                f.write('\n')
        logger.info("save line size:%d to %s" % (count, data_path))


if __name__ == '__main__':
    data_list = parse_data(config.train_path)
    save_data(data_list, config.train_seg_path, stop_words_path=config.stop_words_path)
