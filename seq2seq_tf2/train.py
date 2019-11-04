import os
# from seq2seq_tf2.data_reader import build_dataset, build_test_dataset, load_word_dict, read_vocab, save_word_dict, load_dataset
from seq2seq_tf2.seq2seq_model import Encoder, Decoder, BahdanauAttention
import tensorflow as tf
from seq2seq_tf2 import config

from seq2seq_tf2.seq2seq_model import PGN
from seq2seq_tf2.batcher import batcher, Vocab
from seq2seq_tf2.train_helper import train_model


def train(params):
    assert params["mode"].lower() == "train", "change training mode to 'train'"

    print("Building the model ...")
    model = PGN(params)

    print("Creating the vocab ...")
    vocab = Vocab(params["vocab_path"], params["vocab_size"])
    print('vocab is ', vocab)

    print("Creating the batcher ...")
    b = batcher(params["train_seg_x_dir"], params["train_seg_y_dir"], vocab, params)
    print('b is ', b)

    print("Creating the checkpoint manager")
    checkpoint_dir = "{}/checkpoint".format(params["model_dir"])
    ckpt = tf.train.Checkpoint(step=tf.Variable(0), PGN=model)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=11)

    ckpt.restore(ckpt_manager.latest_checkpoint)
    if ckpt_manager.latest_checkpoint:
        print("Restored from {}".format(ckpt_manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")

    print("Starting the training ...")
    train_model(model, b, params, ckpt, ckpt_manager)


if __name__ == '__main__':
    pass

    # _train(src_vocab_size=config.src_vocab_size,
    #       target_vocab_size=config.target_vocab_size,
    #       embedding_dim=config.embedding_dim,
    #       hidden_dim=config.hidden_dim,
    #       batch_sz=config.batch_sz,
    #       learning_rate=config.learning_rate,
    #       log_dir=config.log_dir,
    #       train_path=config.train_seg_path,
    #       dataset_size=config.dataset_size,
    #       epochs=config.epochs,
    #       steps_per_epoch=config.steps_per_epoch,
    #       checkpoint_path=config.checkpoint_path)
