import tensorflow as tf
from seq2seq_pgn_tf2.models.sequence_to_sequence import SequenceToSequence
from seq2seq_pgn_tf2.models.pgn import PGN
from seq2seq_pgn_tf2.batcher import batcher, Vocab
from seq2seq_pgn_tf2.train_helper import train_model


def train(params):
    assert params["mode"].lower() == "train", "change training mode to 'train'"

    vocab = Vocab(params["vocab_path"], params["vocab_size"])
    print('true vocab is ', vocab)

    print("Creating the batcher ...")
    b = batcher(vocab, params)

    print("Building the model ...")
    if params["model"] == "SequenceToSequence":
        model = SequenceToSequence(params)
    elif params["model"] == "PGN":
        model = PGN(params)

    print("Creating the vocab ...")
    vocab = Vocab(params["vocab_path"], params["vocab_size"])

    print("Creating the checkpoint manager")
    if params["model"] == "SequenceToSequence":
        checkpoint_dir = "{}/checkpoint".format(params["seq2seq_model_dir"])
        ckpt = tf.train.Checkpoint(step=tf.Variable(0), SequenceToSequence=model)
    elif params["model"] == "PGN":
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


if __name__ == '__main__':
    pass

