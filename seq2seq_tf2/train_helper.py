import tensorflow as tf
import time


def train_model(model, dataset, params, ckpt, ckpt_manager):
    optimizer = tf.keras.optimizers.Adagrad(params['learning_rate'],
                                            initial_accumulator_value=params['adagrad_init_acc'],
                                            clipnorm=params['max_grad_norm'])
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

    def loss_function(real, pred, padding_mask):
        loss = 0
        for t in range(real.shape[1]):
            # print('real[:, t]', real[:, t])
            # print('pred[:, t, :]', pred[:, t])
            loss_ = loss_object(real[:, t], pred[:, t])
            mask = tf.cast(padding_mask[:, t], dtype=loss_.dtype)
            mask = tf.cast(mask, dtype=loss_.dtype)
            loss_ *= mask
            loss_ = tf.reduce_mean(loss_)
            loss += loss_
        # print('loss is ', loss)
        # print('real.shape[1] is ', loss / real.shape[1])
        return loss / real.shape[1]

    def _mask_and_avg(values, padding_mask):
        """Applies mask to values then returns overall average (a scalar)
        Args:
          values: a list length max_dec_steps containing arrays shape (batch_size).
          padding_mask: tensor shape (batch_size, max_dec_steps) containing 1s and 0s.
        Returns:
          a scalar
        """
        # padding_mask is Tensor("Cast_2:0", shape=(64, 400), dtype=float32)
        padding_mask = tf.cast(padding_mask, dtype=values[0].dtype)
        dec_lens = tf.reduce_sum(padding_mask, axis=1)  # shape batch_size. float32
        values_per_step = [v * padding_mask[:, dec_step] for dec_step, v in enumerate(values)]
        values_per_ex = sum(values_per_step) / dec_lens  # shape (batch_size); normalized value for each batch member
        # print('tf.reduce_mean(values_per_ex) is ', tf.reduce_mean(values_per_ex))
        return tf.reduce_mean(values_per_ex)  # overall average

    def _coverage_loss(attn_dists, padding_mask):
        """Calculates the coverage loss from the attention distributions.
        Args:
          attn_dists: The attention distributions for each decoder timestep.
          A list length max_dec_steps containing shape (batch_size, attn_length)
          padding_mask: shape (batch_size, max_dec_steps).
        Returns:
          coverage_loss: scalar
        """
        coverage = tf.zeros_like(attn_dists[0])  # shape (batch_size, attn_length). Initial coverage is zero.
        # Coverage loss per decoder timestep. Will be list length max_dec_steps containing shape (batch_size).
        covlosses = []
        for a in attn_dists:
            covloss = tf.reduce_sum(tf.minimum(a, coverage), [1])  # calculate the coverage loss for this step
            covlosses.append(covloss)
            coverage += a  # update the coverage vector
        coverage_loss = _mask_and_avg(covlosses, padding_mask)
        # print('coverage_loss is ', coverage_loss)
        return coverage_loss

    @tf.function
    def train_step(enc_inp, enc_extended_inp, dec_inp, dec_tar, batch_oov_len, enc_padding_mask, padding_mask, cov_loss_wt):
        loss = 0
        with tf.GradientTape() as tape:
            enc_hidden, enc_output = model.call_encoder(enc_inp)
            # print('enc_output is ', enc_output)
            # print('enc_hidden is ', enc_hidden)
            # print('enc_inp is ', enc_inp)
            # print('enc_extended_inp is ', enc_extended_inp)
            # print('dec_inp is ', dec_inp)
            # print('batch_oov_len is ', batch_oov_len)
            # print('enc_padding_mask is ', enc_padding_mask)
            # print('1111111111111111111111111111111111111')
            predictions, _, attentions, coverages = model(enc_output, enc_hidden, enc_inp, enc_extended_inp,
                                                          dec_inp, batch_oov_len, enc_padding_mask,
                                                          params['is_coverage'], prev_coverage=None)
            if params["is_coverage"]:
                # print('oss_function(dec_tar, predictions, padding_mask)', loss_function(dec_tar, predictions, padding_mask))
                # print('cov_loss_wt * _coverage_loss(attentions, enc_padding_mask) is ', cov_loss_wt * _coverage_loss(attentions, enc_padding_mask))
                loss = loss_function(dec_tar, predictions, padding_mask) + cov_loss_wt * _coverage_loss(attentions, enc_padding_mask)
                # print('22222222222222222222')
                # print('loss is ', loss)
            else:
                loss = loss_function(dec_tar, predictions, padding_mask)

            variables = model.encoder.trainable_variables +\
                        model.attention.trainable_variables +\
                        model.decoder.trainable_variables +\
                        model.pointer.trainable_variables
            # print('33333333333')
            gradients = tape.gradient(loss, variables)
            # print('44444444')
            optimizer.apply_gradients(zip(gradients, variables))
            # print('55555555555')
            return loss

    for epoch in range(params["epochs"]):
        t0 = time.time()
        total_loss = 0
        step = 0
        for batch in dataset:
            loss = train_step(batch[0]["enc_input"],
                              batch[0]["extended_enc_input"],
                              batch[1]["dec_input"],
                              batch[1]["dec_target"],
                              batch[0]["max_oov_len"],
                              batch[0]["sample_encoder_pad_mask"],
                              batch[1]["sample_decoder_pad_mask"],
                              cov_loss_wt=0.5)

            total_loss += loss

            step += 1

            if step % 100 == 0:
                print('Epoch {} Batch {} loss {:.4f}'.format(epoch + 1, step, loss.numpy()))

                if step > params['max_steps']:
                    break

        # saving (checkpoint) the model every 1 epochs
        if (epoch + 1) % 1 == 0:
            ckpt_manager.save(checkpoint_number=int(ckpt.step))
            print('Epoch {} Loss {:.4f}'.format(epoch + 1, total_loss / step))
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
    #                           0.5)
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
