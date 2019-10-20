import os
from seq2seq_tf2.data_reader import build_dataset, build_test_dataset, load_word_dict, read_vocab, save_word_dict, load_dataset
from seq2seq_tf2.seq2seq_model import Encoder, Decoder, BahdanauAttention
import tensorflow as tf
from seq2seq_tf2 import config

from seq2seq_tf2.seq2seq_model import PGN
from seq2seq_tf2.batcher import batcher
from seq2seq_tf2.train_helper import train_model


def train(params):
    assert params["mode"].lower() == "train", "change training mode to 'train'"

    tf.compat.v1.logging.info("Building the model ...")
    model = PGN(params)

    tf.compat.v1.logging.info("Creating the batcher ...")
    b = batcher(params["data_dir"], params["vocab_path"], params)

    tf.compat.v1.logging.info("Creating the checkpoint manager")
    logdir = "{}/logdir".format(params["model_dir"])
    checkpoint_dir = "{}/checkpoint".format(params["model_dir"])
    ckpt = tf.train.Checkpoint(step=tf.Variable(0), PGN=model)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=11)

    ckpt.restore(ckpt_manager.latest_checkpoint)
    if ckpt_manager.latest_checkpoint:
        print("Restored from {}".format(ckpt_manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")

    tf.compat.v1.logging.info("Starting the training ...")
    train_model(model, b, params, ckpt, ckpt_manager)


def _train(src_vocab_size='', target_vocab_size='', embedding_dim='', hidden_dim='', batch_sz='',
          learning_rate='', log_dir='', train_path='', dataset_size='', epochs='',
          steps_per_epoch='', checkpoint_path=''):
    encoder = Encoder(vocab_size=src_vocab_size,
                      embedding_dim=embedding_dim,
                      enc_units=hidden_dim,
                      batch_sz=batch_sz)

    decoder = Decoder(vocab_size=target_vocab_size,
                      embedding_dim=embedding_dim,
                      dec_units=hidden_dim,
                      batch_sz=batch_sz)

    optimizer = tf.keras.optimizers.Adam(learning_rate)
    cce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                        reduction=tf.keras.losses.Reduction.NONE)

    def loss_function(real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = cce(real, pred)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        return tf.reduce_mean(loss_)

    checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)

    # log
    file_writer = tf.summary.create_file_writer(logdir=log_dir)

    # data set API
    input_tensor, target_tensor, inp_lang, targ_lang = load_dataset(path=train_path,
                                                                    num_examples=None)
    print(inp_lang.word_index)
    print(len(inp_lang.word_index))
    print(len(targ_lang.word_index))
    dataset = tf.data.Dataset.from_tensor_slices((input_tensor, target_tensor)).shuffle(dataset_size)
    dataset = dataset.batch(batch_sz, drop_remainder=True)

    def train_step(inp, targ, init_enc_hidden):
        loss = 0.0
        with tf.GradientTape() as tape:
            # encoder
            enc_output, enc_hidden = encoder(inp, init_enc_hidden)

            # decoder
            dec_hidden = enc_hidden
            dec_input = tf.expand_dims([targ_lang.word_index['<start>']] * batch_sz, 1)
            # print("dec_input:\n",dec_input)

            # Teacher forcing - feeding the target as the next input
            for t in range(1, targ.shape[1]):
                predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
                # print("predictions:\n", predictions)

                loss += loss_function(targ[:, t], predictions)
                # using teacher forcing
                dec_input = tf.expand_dims(targ[:, t], 1)

        batch_loss = (loss / int(targ.shape[1]))
        variables = encoder.trainable_variables + decoder.trainable_variables
        gradients = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))
        return batch_loss

    for epoch in range(epochs):
        total_loss = 0
        init_enc_hidden = encoder.initialize_hidden_state()

        for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
            batch_loss = train_step(inp, targ, init_enc_hidden)
            total_loss += batch_loss
            if batch % 100 == 0:
                print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1, batch, batch_loss.numpy()))
            # saving (checkpoint) the model every 2 epochs
        if (epoch + 1) % 2 == 0:
            checkpoint.save(file_prefix=checkpoint_path)
        print('Epoch {} Loss {:.4f}'.format(epoch + 1, total_loss / steps_per_epoch))


if __name__ == '__main__':
    _train(src_vocab_size=config.src_vocab_size,
          target_vocab_size=config.target_vocab_size,
          embedding_dim=config.embedding_dim,
          hidden_dim=config.hidden_dim,
          batch_sz=config.batch_sz,
          learning_rate=config.learning_rate,
          log_dir=config.log_dir,
          train_path=config.train_seg_path,
          dataset_size=config.dataset_size,
          epochs=config.epochs,
          steps_per_epoch=config.steps_per_epoch,
          checkpoint_path=config.checkpoint_path)
