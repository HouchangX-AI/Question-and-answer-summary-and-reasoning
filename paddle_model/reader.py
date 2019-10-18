import pandas as pd
import config

start_token = u"<s>"
end_token = u"<e>"
unk_token = u"<unk>"

max_question_len = 100
max_dialogue_len = 800
max_report_len = 100

# def data_reader(path, col_sep='\t'):
#     contents, labels = [], []
#     with open(path, mode='r', encoding='utf-8') as f:
#         for line in f:
#             line = line.strip()
#             if col_sep in line:
#                 index = line.index(col_sep)
#                 label = line[:index].strip()
#                 labels.append(label)
#                 content = line[index + 1:].strip()
#             else:
#                 content = line
#             contents.append(content)
#     return contents, labels


def read_data(path):
    df = pd.read_csv(path, encoding='utf-8')
    question_lens = df['Question'].apply(lambda x: len(x.split(" ")))
    dialogue_lens = df['Dialogue'].apply(lambda x: len(x.split(" ")))
    report_lens = df['Report'].apply(lambda x: len(x.split(" ")))
    data = []
    for i in range(len(df)):
        if question_lens[i] > max_question_len or dialogue_lens[i] > max_dialogue_len or report_lens[i] > max_report_len:
            continue
        item = df.iloc[i]
        data.append([[start_token] + item['Question'].split(" ") + [end_token],
                     [start_token] + item['Dialogue'].split(" ") + [end_token],
                     [start_token] + item['Report'].split(" ") + [end_token]])
    return data


def read_test_data(path):
    df = pd.read_csv(path, encoding='utf-8')
    data = []
    for i in range(len(df)):
        item = df.iloc[i]
        question_vec = item['Question'].split(" ")[0:max_question_len]
        dialogue_vec = item['Dialogue'].split(" ")[0:max_dialogue_len]
        data.append([[start_token] + question_vec + [end_token],
                     [start_token] + dialogue_vec + [end_token]])
    return data


# def read_samples_by_string(path):
#     train = pd.read_csv(path, encoding='utf-8')
#     lines = []
#     for k in ['Question', 'Dialogue', 'Report']:
#         train_values = list(train[k].values)
#         lines.extend(train_values)
#
#     for line in lines:
#         line = line.strip()
#         if not line:
#             continue
#         parts = line.lower().strip()
#         yield parts


def build_dataset(path):
    print('Read data, path:{0}'.format(path))
    train = pd.read_csv(path, encoding='utf-8')
    lines = []
    for k in ['Question', 'Dialogue', 'Report']:
        lines.extend(list(train[k].values))

    return lines

# word_lst = []
#     for i in data_content:
#         word_lst.extend(i.split())