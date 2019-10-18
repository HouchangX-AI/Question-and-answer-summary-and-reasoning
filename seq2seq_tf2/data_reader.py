from collections import Counter
import io
import tensorflow as tf
from seq2seq_tf2 import config
# Define constants associated with the usual special tokens.
PAD_TOKEN = 'PAD'
GO_TOKEN = 'GO'
EOS_TOKEN = 'EOS'
UNK_TOKEN = 'UNK'


def load_word_dict(save_path):
    dict_data = dict()
    with open(save_path, 'r', encoding='utf-8') as f:
        for line in f:
            items = line.strip().split()
            try:
                dict_data[items[0]] = int(items[1])
            except IndexError:
                print('error', line)
    return dict_data


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


def read_samples_by_string(path):
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.lower().strip().split('\t')
            if len(parts) == 3:
                question, dialogue, report = parts[0], parts[1], parts[2]
                yield question, dialogue, report
            if len(parts) == 2:
                question, dialogue = parts[0], parts[1]
                yield question, dialogue


def build_dataset(path):
    print('Read data, path:{0}'.format(path))
    question, dialogue, report = [], [], []
    for q, d, r in read_samples_by_string(path):
        question.append(q)
        dialogue.append(d)
        report.append(r)
    return question, dialogue, report


def build_test_dataset(path):
    print('Read data, path:{0}'.format(path))
    question, dialogue = [], []
    for q, d in read_samples_by_string(path):
        question.append(q)
        dialogue.append(d)
    return question, dialogue


def preprocess_sentence(w):
    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    # w = re.sub(r"([?.!,¿])", r" \1 ", w)
    # w = re.sub(r'[" "]+', " ", w)

    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    # w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)

    # w = w.rstrip().strip()

    # adding a start and an end token to the sentence
    # so that the model know when to start and stop predicting.
    w = '<start> ' + w + ' <end>'
    return w


def create_dataset(path, num_examples):
    lines = io.open(path, encoding='UTF-8').read().strip().split('\n')
    word_pairs = [[preprocess_sentence(w) for w in l.split('\t')[:-1]] for l in lines[:num_examples]]
    question, dialogue, report = [], [], []
    for i in word_pairs:
        question.append(i[0])
        dialogue.append(i[1])
        if len(i) == 3:
            report.append(i[2])
        else:
            report.append('<start> ' + 'none' + ' <end>')

    # report = [i[2] for i in word_pairs]
    return question, dialogue, report


def load_dataset(path, num_examples=None):
    # creating cleaned input, output pairs
    # question, dialogue, report = build_dataset(path)
    # inp_lang = question + dialogue
    # targ_lang = report
    # targ_lang, inp_lang = create_dataset(path, num_examples)
    question, dialogue, report = create_dataset(path, num_examples)
    # print(question, dialogue, report)
    # targ_lang, inp_lang = create_dataset(path, num_examples)
    input_tensor, inp_lang_tokenizer = tokenize(dialogue)
    target_tensor, targ_lang_tokenizer = tokenize(report)

    return input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer


def tokenize(lang):
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
    lang_tokenizer.fit_on_texts(lang)
    tensor = lang_tokenizer.texts_to_sequences(lang)
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')
    return tensor, lang_tokenizer


if __name__ == '__main__':
    input_tensor, target_tensor, inp_lang, targ_lang = load_dataset(path=config.train_seg_path, num_examples=None)
    print("input_tensor:\n", input_tensor)
    print("target_tensor:\n", target_tensor)