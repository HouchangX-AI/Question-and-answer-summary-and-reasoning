import os
import pathlib


# pwd_path = os.path.abspath(os.path.dirname(__file__))
pwd_path = pathlib.Path(os.path.abspath(__file__)).parent.parent

output_dir = os.path.join(pwd_path, 'datasets')
# Training data path.
train_path = os.path.join(output_dir, 'AutoMaster_TrainSet.csv')
# Validation data path.
test_path = os.path.join(output_dir, 'AutoMaster_TestSet.csv')

# paddle_train config
save_vocab_path = os.path.join(output_dir, 'vocab.txt')
model_save_dir = os.path.join(output_dir, 'paddle_model')

vocab_max_size = 7000
vocab_min_count = 5

use_cuda = False

batch_size = 64
rnn_hidden_dim = 128
maxlen = 400
dropout = 0.0
gpu_id = 0
# segment of train file
train_seg_path_x = os.path.join(output_dir, 'train_set.seg_x.txt')
train_seg_path_y = os.path.join(output_dir, 'train_set.seg_y.txt')
test_seg_path_x = os.path.join(output_dir, 'test_set.seg_x.txt')
# segment of test file
test_seg_path = os.path.join(output_dir, 'test_set.seg.csv')

stop_words_path = os.path.join(output_dir, 'stop_words.txt')
sentence_path = os.path.join(output_dir, 'sentences.txt')
word2vec_output = os.path.join(output_dir, 'word2vec.txt')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

src_vocab_size = 2377 + 1
target_vocab_size = 905 + 1
embedding_dim = 256
hidden_dim = 512
batch_sz = 16
learning_rate = 0.001
log_dir = "./log/"
# train_path
dataset_size = 100
epochs = 10
steps_per_epoch = dataset_size // batch_sz
checkpoint_path = "./checkpoints/lstm"

vocab_path = os.path.join(output_dir, 'vocab.txt')
beam_size = 3

