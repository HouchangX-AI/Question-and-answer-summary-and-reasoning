import tensorflow as tf
import argparse
import os
from seq2seq_tf2.train import train
from utils.log_utils import define_logger
from seq2seq_tf2 import config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_enc_len", default=400, help="Encoder input max sequence length", type=int)
    parser.add_argument("--max_dec_len", default=100, help="Decoder input max sequence length", type=int)
    parser.add_argument("--batch_size", default=16, help="batch size", type=int)
    parser.add_argument("--vocab_size", default=50000, help="Vocabulary size", type=int)
    parser.add_argument("--embed_size", default=128, help="Words embeddings dimension", type=int)
    parser.add_argument("--enc_units", default=256, help="Encoder GRU cell units number", type=int)
    parser.add_argument("--dec_units", default=256, help="Decoder GRU cell units number", type=int)
    parser.add_argument("--attn_units", default=512, help="[context vector, decoder state, decoder input] feedforward result dimension - this result is used to compute the attention weights", type=int)
    parser.add_argument("--learning_rate", default=0.15, help="Learning rate", type=float)
    parser.add_argument("--adagrad_init_acc", default=0.1, help="Adagrad optimizer initial accumulator value. Please refer to the Adagrad optimizer API documentation on tensorflow site for more details.", type=float)
    parser.add_argument("--max_grad_norm",default=0.8, help="Gradient norm above which gradients must be clipped", type=float)
    parser.add_argument("--checkpoints_save_steps", default=10000, help="Save checkpoints every N steps", type=int)
    parser.add_argument("--mode", help="training, eval or test options")
    parser.add_argument("--model_dir", help="Model folder")
    parser.add_argument("--data_dir_1",  help="Data Folder")
    parser.add_argument("--data_dir_2", help="Data Folder")
    parser.add_argument("--vocab_path", help="Vocab path")
    parser.add_argument("--log_file", help="File in which to redirect console outputs", default="", type=str)

    args = parser.parse_args()
    params = vars(args)
    params["mode"] = "train"
    params["data_dir_1"] = config.train_seg_path_x
    params["data_dir_2"] = config.train_seg_path_y
    params["vocab_path"] = config.vocab_path
    print(params)

    assert params["mode"], "mode is required. train, test or eval option"
    assert params["mode"] in ["train", "test", "eval"], "The mode must be train , test or eval"
    # assert os.path.exists(params["data_dir"]), "data_dir doesn't exist"
    # assert os.path.isfile(params["vocab_path"]), "vocab_path doesn't exist"

    if not os.path.exists("{}".format(params["model_dir"])):
        os.makedirs("{}".format(params["model_dir"]))
    """i = len([name for name in os.listdir("{}/{}".format(params["model_dir"], "logdir")) if os.path.isfile(name)])
    params["log_file"] = "{}/logdir/tensorflow_{}.log".format(params["model_dir"],i)"""

    if params["mode"] == "train":
        train(params)


if __name__ == '__main__':
    main()
