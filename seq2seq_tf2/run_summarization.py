import tensorflow as tf
import argparse
from seq2seq_tf2.train import train
from seq2seq_tf2.test import test_and_save
from utils.log_utils import define_logger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_enc_len", default=400, help="Encoder input max sequence length", type=int)
    parser.add_argument("--max_dec_len", default=100, help="Decoder input max sequence length", type=int)
    parser.add_argument("--max_dec_steps", default=120, help="maximum number of words of the predicted abstract", type=int)
    parser.add_argument("--min_dec_steps", default=30, help="Minimum number of words of the predicted abstract", type=int)
    parser.add_argument("--batch_size", default=3, help="batch size", type=int)
    parser.add_argument("--beam_size", default=3,
                        help="beam size for beam search decoding (must be equal to batch size in decode mode)",
                        type=int)
    parser.add_argument("--vocab_size", default=50000, help="Vocabulary size", type=int)
    parser.add_argument("--embed_size", default=256, help="Words embeddings dimension", type=int)
    parser.add_argument("--enc_units", default=256, help="Encoder GRU cell units number", type=int)
    parser.add_argument("--dec_units", default=256, help="Decoder GRU cell units number", type=int)
    parser.add_argument("--attn_units", default=256,
                        help="[context vector, decoder state, decoder input] feedforward result dimension - "
                             "this result is used to compute the attention weights", type=int)
    parser.add_argument("--learning_rate", default=0.15, help="Learning rate", type=float)
    parser.add_argument("--adagrad_init_acc", default=0.1,
                        help="Adagrad optimizer initial accumulator value. Please refer to the Adagrad optimizer "
                             "API documentation on tensorflow site for more details.", type=float)
    parser.add_argument("--max_grad_norm", default=0.8, help="Gradient norm above which gradients must be clipped",
                        type=float)
    parser.add_argument("--checkpoints_save_steps", default=10, help="Save checkpoints every N steps", type=int)
    parser.add_argument("--max_steps", default=10000, help="Max number of iterations", type=int)
    parser.add_argument("--num_to_test", default=5, help="Number of examples to test", type=int)
    parser.add_argument("--mode", default='test', help="training, eval or test options")
    parser.add_argument("--pointer_gen", default=True, help="training, eval or test options")
    parser.add_argument("--is_coverage", default=True, help="is_coverage")
    parser.add_argument("--model_dir", default='./ckpt', help="Model folder")
    parser.add_argument("--model_path", help="Path to a specific model", default="", type=str)
    parser.add_argument("--train_seg_x_dir", default='../datasets/train_set.seg_x.txt', help="train_seg_x_dir")
    parser.add_argument("--train_seg_y_dir", default='../datasets/train_set.seg_y.txt', help="train_seg_y_dir")
    parser.add_argument("--test_seg_x_dir", default='../datasets/test_set.seg_x.txt', help="test_seg_x_dir")
    parser.add_argument("--vocab_path", default='../datasets/vocab.txt', help="Vocab path")
    parser.add_argument("--word2vec_output", default='../datasets/word2vec.txt', help="Vocab path")
    parser.add_argument("--log_file", help="File in which to redirect console outputs", default="", type=str)
    parser.add_argument("--test_save_dir", default='../datasets/', help="test_save_dir")

    args = parser.parse_args()
    params = vars(args)

    if params["mode"] == "train":
        train(params)

    elif params["mode"] == "test":
        test_and_save(params)


if __name__ == '__main__':
    main()
