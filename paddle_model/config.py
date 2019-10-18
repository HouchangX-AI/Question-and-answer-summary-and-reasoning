import os
import pathlib


# pwd_path = os.path.abspath(os.path.dirname(__file__))
pwd_path = pathlib.Path(os.path.abspath(__file__)).parent.parent

# Training data path.
# chinese corpus
raw_train_paths = [
    # os.path.join(pwd_path, '../data/cn/CGED/CGED18_HSK_TrainingSet.xml'),
    # os.path.join(pwd_path, '../data/cn/CGED/CGED17_HSK_TrainingSet.xml'),
    # os.path.join(pwd_path, '../data/cn/CGED/CGED16_HSK_TrainingSet.xml'),
    os.path.join(pwd_path, '../data/cn/CGED/sample_HSK_TrainingSet.xml'),
]

output_dir = os.path.join(pwd_path, 'datasets')
# Training data path.
train_path = os.path.join(output_dir, 'AutoMaster_TrainSet.csv')
# Validation data path.
test_path = os.path.join(output_dir, 'AutoMaster_TestSet.csv')

# paddle_train config
save_vocab_path = os.path.join(output_dir, 'vocab.txt')
model_save_dir = os.path.join(output_dir, 'paddle_model')

vocab_max_size = 5000
vocab_min_count = 5
hidden_dim = 512

use_cuda = False

batch_size = 64
epochs = 40
rnn_hidden_dim = 128
maxlen = 400
dropout = 0.0
gpu_id = 0
# segment of train file
train_seg_path = os.path.join(output_dir, 'train_set.seg.csv')
# segment of test file
test_seg_path = os.path.join(output_dir, 'test_set.seg.csv')

stop_words_path = os.path.join(output_dir, 'stop_words.txt')


if not os.path.exists(output_dir):
    os.makedirs(output_dir)
