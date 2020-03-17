import tensorflow as tf
from seq2seq_pgn_tf2.models.sequence_to_sequence import SequenceToSequence
from seq2seq_pgn_tf2.models.pgn import PGN
from seq2seq_pgn_tf2.batcher import Vocab, batcher
from seq2seq_pgn_tf2.test_helper import beam_decode, batch_greedy_decode
from tqdm import tqdm


def test(params):
    assert params["mode"].lower() == "test", "change training mode to 'test' or 'eval'"
    assert params["beam_size"] == params["batch_size"], "Beam size must be equal to batch_size, change the params"

    print("Building the model ...")
    if params["model"] == "SequenceToSequence":
        model = SequenceToSequence(params)
    elif params["model"] == "PGN":
        model = PGN(params)

    print("Creating the vocab ...")
    vocab = Vocab(params["vocab_path"], params["vocab_size"])

    print("Creating the batcher ...")
    b = batcher(vocab, params)

    print("Creating the checkpoint manager")
    if params["model"] == "SequenceToSequence":
        checkpoint_dir = "{}/checkpoint".format(params["seq2seq_model_dir"])
        ckpt = tf.train.Checkpoint(step=tf.Variable(0), SequenceToSequence=model)
    elif params["model"] == "PGN":
        checkpoint_dir = "{}/checkpoint".format(params["pgn_model_dir"])
        ckpt = tf.train.Checkpoint(step=tf.Variable(0), PGN=model)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=5)

    # path = params["model_path"] if params["model_path"] else ckpt_manager.latest_checkpoint
    # path = ckpt_manager.latest_checkpoint
    # print('path is ', path)
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print("Model restored")
    for batch in b:
        yield batch_greedy_decode(model, batch, vocab, params)


def test_and_save(params):
    assert params["test_save_dir"], "provide a dir where to save the results"
    gen = test(params)
    with tqdm(total=params["num_to_test"], position=0, leave=True) as pbar:
        for i in range(params["num_to_test"]):
            trial = next(gen)
            with open(params["test_save_dir"] + "/article_" + str(i) + ".txt", "w", encoding='utf-8') as f:
                f.write("article:\n")
                f.write(trial.text)
                f.write("\n\nabstract:\n")
                f.write(trial.abstract)
            pbar.update(1)


if __name__ == '__main__':
    test('我的帕萨特烧机油怎么办')

