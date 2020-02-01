import tensorflow as tf
import time
from seq2seq_pgn_tf2.loss import loss_function


def train_model(model, dataset, params, ckpt, ckpt_manager):
    # optimizer = tf.keras.optimizers.Adagrad(params['learning_rate'],
    #                                         initial_accumulator_value=params['adagrad_init_acc'],
    #                                         clipnorm=params['max_grad_norm'])
    optimizer = tf.keras.optimizers.Adam(name='Adam', learning_rate=params["learning_rate"])

    # @tf.function
    def train_step(enc_inp, enc_extended_inp, dec_inp, dec_tar, batch_oov_len, enc_padding_mask, padding_mask):
        # loss = 0
        with tf.GradientTape() as tape:
            enc_output, enc_hidden = model.call_encoder(enc_inp)
            dec_hidden = enc_hidden
            predictions, _, attentions, coverages = model(enc_output,  # shape=(3, 200, 256)
                                                          dec_hidden,  # shape=(3, 256)
                                                          enc_inp,  # shape=(3, 200)
                                                          enc_extended_inp,  # shape=(3, 200)
                                                          dec_inp,  # shape=(3, 50)
                                                          batch_oov_len,  # shape=()
                                                          enc_padding_mask,  # shape=(3, 200)
                                                          params['is_coverage'],
                                                          prev_coverage=None)

            loss = loss_function(dec_tar,
                                 predictions,
                                 padding_mask,
                                 attentions,
                                 params["cov_loss_wt"],
                                 params['is_coverage'])

        variables = model.encoder.trainable_variables +\
                    model.attention.trainable_variables +\
                    model.decoder.trainable_variables +\
                    model.pointer.trainable_variables
        gradients = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))
        return loss

    for epoch in range(params['epochs']):
        t0 = time.time()
        step = 0
        for batch in dataset:
            loss = train_step(batch[0]["enc_input"],  # shape=(16, 200)
                              batch[0]["extended_enc_input"],  # shape=(16, 200)
                              batch[1]["dec_input"],  # shape=(16, 50)
                              batch[1]["dec_target"],  # shape=(16, 50)
                              batch[0]["max_oov_len"],  # ()
                              batch[0]["sample_encoder_pad_mask"],  # shape=(16, 200)
                              batch[1]["sample_decoder_pad_mask"])  # shape=(16, 50)

            step += 1
            if step % 2 == 0:
                print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1, step, loss.numpy()))

        if (epoch + 1) % 1 == 0:
            ckpt_save_path = ckpt_manager.save()
            print('Saving checkpoint for epoch {} at {} ,best loss {}'.format(epoch + 1, ckpt_save_path, loss))
            print('Epoch {} Loss {:.4f}'.format(epoch + 1, loss))
            print('Time taken for 1 epoch {} sec\n'.format(time.time() - t0))


    # try:
    #     for batch in dataset:
    #         t0 = time.time()
    #         loss = train_step(batch[0]["enc_input"],
    #                           batch[0]["extended_enc_input"],
    #                           batch[1]["dec_input"],
    #                           batch[1]["dec_target"],
    #                           batch[0]["max_oov_len"],
    #                           batch[0]["sample_encoder_pad_mask"],
    #                           batch[1]["sample_decoder_pad_mask"],
    #                           )
    #         print('Step {}, time {:.4f}, Loss {:.4f}'.format(int(ckpt.step),
    #                                                          time.time() - t0,
    #                                                          loss.numpy()))
    #         if int(ckpt.step) == params["max_steps"]:
    #             ckpt_manager.save(checkpoint_number=int(ckpt.step))
    #             print("Saved checkpoint for step {}".format(int(ckpt.step)))
    #             break
    #         if int(ckpt.step) % params["checkpoints_save_steps"] == 0:
    #             ckpt_manager.save(checkpoint_number=int(ckpt.step))
    #             print("Saved checkpoint for step {}".format(int(ckpt.step)))
    #         ckpt.step.assign_add(1)
    #
    # except KeyboardInterrupt:
    #     ckpt_manager.save(int(ckpt.step))
    #     print("Saved checkpoint for step {}".format(int(ckpt.step)))
