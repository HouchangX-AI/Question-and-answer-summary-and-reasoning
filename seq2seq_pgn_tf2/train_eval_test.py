import tensorflow as tf
from seq2seq_pgn_tf2.models.pgn import PGN
from seq2seq_pgn_tf2.batcher import batcher, Vocab
from seq2seq_pgn_tf2.train_helper import train_model
from seq2seq_pgn_tf2.test_helper import beam_decode
from tqdm import tqdm
from utils.data_utils import get_result_filename
import pandas as pd
# from rouge import Rouge
import pprint
import numpy as np


def train(params):
    assert params["mode"].lower() == "train", "change training mode to 'train'"

    print("Creating the vocab ...")
    vocab = Vocab(params["vocab_path"], params["vocab_size"])

    print("Creating the batcher ...")
    b = batcher(vocab, params)

    print("Building the model ...")
    model = PGN(params)

    print("Creating the checkpoint manager")
    checkpoint_dir = "{}/checkpoint".format(params["pgn_model_dir"])
    ckpt = tf.train.Checkpoint(step=tf.Variable(0), PGN=model)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=5)

    ckpt.restore(ckpt_manager.latest_checkpoint)
    if ckpt_manager.latest_checkpoint:
        print("Restored from {}".format(ckpt_manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")

    print("Starting the training ...")
    train_model(model, b, params, ckpt, ckpt_manager)


def test(params):
    assert params["mode"].lower() == "test", "change training mode to 'test' or 'eval'"
    # assert params["beam_size"] == params["batch_size"], "Beam size must be equal to batch_size, change the params"

    print("Building the model ...")
    model = PGN(params)

    print("Creating the vocab ...")
    vocab = Vocab(params["vocab_path"], params["vocab_size"])

    print("Creating the batcher ...")
    b = batcher(vocab, params)

    print("Creating the checkpoint manager")
    checkpoint_dir = "{}/checkpoint".format(params["pgn_model_dir"])
    ckpt = tf.train.Checkpoint(PGN=model)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=5)

    # path = params["model_path"] if params["model_path"] else ckpt_manager.latest_checkpoint
    # path = ckpt_manager.latest_checkpoint
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print("Model restored")
    for batch in b:
        yield beam_decode(model, batch, vocab, params)
        

def test_and_save(params):
    assert params["test_save_dir"], "provide a dir where to save the results"
    gen = test(params)
    results = []
    with tqdm(total=params["num_to_test"], position=0, leave=True) as pbar:
        for i in range(params["num_to_test"]):
            trial = next(gen)
            results.append(trial.abstract)
            pbar.update(1)
    return results


def predict_result(params):
    # 预测结果
    results = test_and_save(params)
    # 保存结果
    save_predict_result(results, params)


def save_predict_result(results, params):
    # 读取结果
    test_df = pd.read_csv(params['test_x_dir'])
    # 填充结果
    test_df['Prediction'] = results[:20000]
    # 　提取ID和预测结果两列
    test_df = test_df[['QID', 'Prediction']]
    # 保存结果.
    result_save_path = get_result_filename(params)
    test_df.to_csv(result_save_path, index=None, sep=',')


if __name__ == '__main__':
    pass

