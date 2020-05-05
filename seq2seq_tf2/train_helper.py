import tensorflow as tf
import time

START_DECODING = '[START]'


def train_model(model, dataset, params, ckpt, ckpt_manager):
    # optimizer = tf.keras.optimizers.Adagrad(params['learning_rate'],
    #                                         initial_accumulator_value=params['adagrad_init_acc'],
    #                                         clipnorm=params['max_grad_norm'])
    optimizer = tf.keras.optimizers.Adam(name='Adam', learning_rate=params["learning_rate"])
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction='none')

    # 定义损失函数
    def loss_function(real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 1))
        # dec_lens = tf.reduce_sum(tf.cast(mask, dtype=tf.float32), axis=-1)
        # print('dec_lens is ', dec_lens)
        # print('real is ', real)
        # print('mask is ', mask)
        loss_ = loss_object(real, pred)
        # print('loss_ is ', loss_)
        mask = tf.cast(mask, dtype=loss_.dtype)

        # loss_,mask (batch_size, dec_len-1)
        loss_ *= mask
        # print('88888888888 is ', loss_)
        # loss_ = tf.reduce_sum(loss_, axis=-1)/dec_lens
        # print('tf.reduce_sum(loss_, axis=-1)', tf.reduce_sum(loss_, axis=-1))
        # print('999999999 is ', loss_)
        # print('tf.reduce_mean(loss_) is ', tf.reduce_mean(loss_))
        return tf.reduce_mean(loss_)
    
    # def loss_function(real, pred):
    #     mask = tf.math.logical_not(tf.math.equal(real, pad_index))
    #     loss_ = loss_object(real, pred)
    #     mask = tf.cast(mask, dtype=loss_.dtype)
    #     loss_ *= mask
    #     return tf.reduce_mean(loss_)


    # @tf.function()
    def train_step(enc_inp, dec_tar, dec_inp):
        # loss = 0
        with tf.GradientTape() as tape:
            enc_output, enc_hidden = model.call_encoder(enc_inp)
            dec_hidden = enc_hidden
            # start index
            pred, _ = model(enc_output,  # shape=(3, 200, 256)
                            dec_inp,  # shape=(3, 256)
                            dec_hidden,  # shape=(3, 200)
                            dec_tar)  # shape=(3, 50) 
            loss = loss_function(dec_tar, pred)
                        
        # variables = model.trainable_variables
        variables = model.encoder.trainable_variables + model.attention.trainable_variables + model.decoder.trainable_variables
        gradients = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))
        return loss

    best_loss = 20
    epochs = params['epochs']
    for epoch in range(epochs):
        t0 = time.time()
        step = 0
        total_loss = 0
        # for step, batch in enumerate(dataset.take(params['steps_per_epoch'])):
        for batch in dataset:
        # for batch in dataset.take(params['steps_per_epoch']):
            # if step == 1:
            # print('&&&&&& ', step)
            # print('batch[1]["dec_target"] is ', batch[0]["enc_input"])
            # print('batch[1]["dec_input"] is ', batch[1]["dec_input"])
            loss = train_step(batch[0]["enc_input"],  # shape=(16, 200)
                              batch[1]["dec_target"], # shape=(16, 50)
                              batch[1]["dec_input"])
            # if step == 1:
            # print('loss is ', loss)
            # time.sleep(2)
            # print('loss is ', loss)
            step += 1
            total_loss += loss
            if step % 100 == 0:
                # print('batch[0]["enc_input"] is ', batch[0]["enc_input"])
                # print('batch[1]["dec_target"] is ', batch[1]["dec_target"])
                # print('batch[1]["dec_input"] is ', batch[1]["dec_input"])
                print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1, step, total_loss / step))
                # print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1, step, loss.numpy()))

        if epoch % 1 == 0: 
            if total_loss / step < best_loss:
                best_loss = total_loss / step
                ckpt_save_path = ckpt_manager.save()
                print('Saving checkpoint for epoch {} at {} ,best loss {}'.format(epoch + 1, ckpt_save_path, best_loss))
                print('Epoch {} Loss {:.4f}'.format(epoch + 1, total_loss / step))
                print('Time taken for 1 epoch {} sec\n'.format(time.time() - t0))

