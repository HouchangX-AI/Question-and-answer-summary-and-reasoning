import tensorflow as tf
import time


def train_model(model, dataset, params, ckpt, ckpt_manager):
    optimizer = tf.keras.optimizers.Adagrad(params['learning_rate'],
                                            initial_accumulator_value=params['adagrad_init_acc'],
                                            clipnorm=params['max_grad_norm'])
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction='none')

    def loss_function(real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 1))
        dec_lens = tf.reduce_sum(tf.cast(mask, dtype=tf.float32), axis=-1)
        loss_ = loss_object(real, pred)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        loss_ = tf.reduce_sum(loss_, axis=-1) / dec_lens  # we have to make sure no empty abstract is being used otherwise dec_lens may contain null values
        return tf.reduce_mean(loss_)

    @tf.function(input_signature=(tf.TensorSpec(shape=[params["batch_size"], None], dtype=tf.int32),
                                  tf.TensorSpec(shape=[params["batch_size"], None], dtype=tf.int32),
                                  tf.TensorSpec(shape=[params["batch_size"], params["max_dec_len"]], dtype=tf.int32),
                                  tf.TensorSpec(shape=[params["batch_size"], params["max_dec_len"]], dtype=tf.int32),
                                  tf.TensorSpec(shape=[], dtype=tf.int32)))
    def train_step(enc_inp, enc_extended_inp, dec_inp, dec_tar, batch_oov_len):
        loss = 0

        with tf.GradientTape() as tape:
            enc_hidden, enc_output = model.call_encoder(enc_inp)
            print('enc_hidden is ', enc_hidden)
            print('enc_output is ', enc_output)
            predictions, _ = model(enc_output, enc_hidden, enc_inp, enc_extended_inp, dec_inp, batch_oov_len)
            print('predictions is ', predictions)
            loss = loss_function(dec_tar, predictions)

        variables = model.encoder.trainable_variables + model.attention.trainable_variables + model.decoder.trainable_variables + model.pointer.trainable_variables
        gradients = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))
        return loss

    try:
        for batch in dataset:
            # print("batch is {}".format(batch))
            t0 = time.time()
            print('batch[0]["enc_input"] is ', batch[0]["enc_input"])
            print('batch[0]["extended_enc_input"] is ', batch[0]["extended_enc_input"])
            print('batch[1]["dec_input"] is ', batch[1]["dec_input"])
            print('batch[1]["dec_target"] is ', batch[1]["dec_target"])
            print('batch[0]["max_oov_len"] is ', batch[0]["max_oov_len"])
            loss = train_step(batch[0]["enc_input"], batch[0]["extended_enc_input"], batch[1]["dec_input"],
                              batch[1]["dec_target"], batch[0]["max_oov_len"])
            print('Step {}, time {:.4f}, Loss {:.4f}'.format(int(ckpt.step),
                                                             time.time() - t0,
                                                             loss.numpy()))
            if int(ckpt.step) == params["max_steps"]:
                ckpt_manager.save(checkpoint_number=int(ckpt.step))
                print("Saved checkpoint for step {}".format(int(ckpt.step)))
                break
            if int(ckpt.step) % params["checkpoints_save_steps"] == 0:
                ckpt_manager.save(checkpoint_number=int(ckpt.step))
                print("Saved checkpoint for step {}".format(int(ckpt.step)))
            ckpt.step.assign_add(1)

    except KeyboardInterrupt:
        ckpt_manager.save(int(ckpt.step))
        print("Saved checkpoint for step {}".format(int(ckpt.step)))
