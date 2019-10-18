import config
import pandas as pd
import jieba
from jieba import posseg


def read_stopwords(path):
    lines = set()
    with open(path, mode='r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            lines.add(line)
    return lines

# def seg_data(in_file, out_file, col_sep='\t', stop_words_path=''):
#     stopwords = read_stopwords(stop_words_path)
#     with open(in_file, 'r', encoding='utf-8') as f1, open(out_file, 'w', encoding='utf-8') as f2:
#         count = 0
#         for line in f1:
#             line = line.rstrip()
#             parts = line.split(col_sep)
#             if len(parts) < 2:
#                 continue
#             label = parts[0].strip()
#             data = ' '.join(parts[1:])
#             seg_list = jieba.lcut(data)
#             seg_words = []
#             for i in seg_list:
#                 if i in stopwords:
#                     continue
#                 seg_words.append(i)
#             seg_line = ' '.join(seg_words)
#             if count % 10000 == 0:
#                 logger.info('count:%d' % count)
#                 logger.info(line)
#                 logger.info('=' * 20)
#                 logger.info(seg_line)
#             count += 1
#             f2.write('%s\t%s\n' % (label, seg_line))
#         logger.info('%s to %s, size: %d' % (in_file, out_file, count))


def seg_line(line):
    stopwords = read_stopwords(config.stop_words_path)
    tokens = posseg.cut(line)
    result = []
    for i, j in tokens:
        if j == 'x':
            continue
        if i in stopwords:
            continue
        result.append(i)
    return " ".join(result)


def proc(line):
    if isinstance(line, str):
        tokens = line.split("|")
        seg_list = []
        for t in tokens:
            seg_list.append(seg_line(t))
        return " | ".join(seg_list)


def seg_data(train_path, test_path):
    # load original data
    train = pd.read_csv(train_path)[:200]
    test = pd.read_csv(test_path)[:40]

    # segmentation
    for k in ['Brand', 'Model', 'Question', 'Dialogue']:
        train[k] = train[k].apply(proc)
        test[k] = test[k].apply(proc)

    train['Report'] = train['Report'].apply(proc)

    # save segmented data
    train.to_csv(config.train_seg_path, index=False, encoding='utf-8')
    test.to_csv(config.test_seg_path, index=False, encoding='utf-8')


