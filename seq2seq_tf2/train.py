import tensorflow as tf
from seq2seq_tf2.seq2seq_model import PGN
from seq2seq_tf2.batcher import batcher, Vocab
from seq2seq_tf2.train_helper import train_model


def train(params):
    assert params["mode"].lower() == "train", "change training mode to 'train'"

    print("Building the model ...")
    model = PGN(params)

    print("Creating the vocab ...")
    vocab = Vocab(params["vocab_path"], params["vocab_size"])
    print('true vocab is ', vocab)

    print("Creating the batcher ...")
    b = batcher(vocab, params)

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

