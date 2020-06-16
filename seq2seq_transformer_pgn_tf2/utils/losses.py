import tensorflow as tf


loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction='none')


def loss_function(real, outputs, padding_mask, cov_loss_wt, use_coverage):
    pred = outputs["logits"]
    attn_dists = outputs["attentions"]
    if use_coverage:
        loss = pgn_log_loss_function(real, pred, padding_mask) + cov_loss_wt * _coverage_loss(attn_dists, padding_mask)
        return loss
    else:
        return seq2seq_loss_function(real, pred, padding_mask)


def seq2seq_loss_function(real, pred, padding_mask):
    """
    跑seq2seq时用的Loss
    :param real: shape=(16, 50)
    :param pred: shape=(16, 50, 30000)
    :return:
    """
    loss = 0
    for t in range(real.shape[1]):
        loss_ = loss_object(real[:, t], pred[:, t])
        mask = tf.cast(padding_mask[:, t], dtype=loss_.dtype)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        loss_ = tf.reduce_mean(loss_)
        loss += loss_
    return loss / real.shape[1]


def pgn_log_loss_function(real, final_dists, padding_mask):
    # Calculate the loss per step
    # This is fiddly; we use tf.gather_nd to pick out the probabilities of the gold target words
    loss_per_step = []  # will be list length max_dec_steps containing shape (batch_size)
    batch_nums = tf.range(0, limit=real.shape[0])  # shape (batch_size)
    for dec_step, dist in enumerate(final_dists):
        # The indices of the target words. shape (batch_size)
        targets = real[:, dec_step]
        indices = tf.stack((batch_nums, targets), axis=1)  # shape (batch_size, 2)
        gold_probs = tf.gather_nd(dist, indices)  # shape (batch_size). prob of correct words on this step
        losses = -tf.math.log(gold_probs)
        loss_per_step.append(losses)
    # Apply dec_padding_mask and get loss
    # print('loss_per_step is ', loss_per_step)
    _loss = _mask_and_avg(loss_per_step, padding_mask)
    return _loss


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
    # print('values is ', values)
    values_per_step = [v * padding_mask[:, dec_step] for dec_step, v in enumerate(values)]
    values_per_ex = sum(values_per_step) / dec_lens  # shape (batch_size); normalized value for each batch member
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
        print('a is ', a)
        covloss = tf.reduce_sum(tf.minimum(a, coverage), [1])  # calculate the coverage loss for this step
        covlosses.append(covloss)
        coverage += a  # update the coverage vector
    coverage_loss = _mask_and_avg(covlosses, padding_mask)
    return coverage_loss