import tensorflow as tf
from seq2seq_tf2.seq2seq_model import PGN
from seq2seq_tf2.batcher import Vocab, START_DECODING, STOP_DECODING, article_to_ids, output_to_words, SENTENCE_END


def test(params):
    assert params["mode"].lower() == "test", "change mode to 'test'"

    # with open(data_path_3, 'w', encoding='utf-8') as f3:
    #     for line in data_3:
    #         if isinstance(line, str):
    #             seg_list = segment(line.strip(), cut_type='word')
    #             seg_line = ' '.join(seg_list)
    #             f3.write('%s' % seg_line)
    #         f3.write('\n')
    # with open(params["test_seg_x_dir"], 'r', encoding='utf-8') as f:
    #     for
    model = PGN(params)

    ckpt = tf.train.Checkpoint(model=model)
    checkpoint_dir = "{}/checkpoint".format(params["model_dir"])
    latest_ckpt = tf.train.latest_checkpoint(checkpoint_dir)

    # manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=3)
    # ckpt.restore(manager.latest_checkpoint)
    ckpt.restore(latest_ckpt)

    vocab = Vocab(params["vocab_path"], params["vocab_size"])

    input_datasets = tf.data.TextLineDataset(params["test_seg_x_dir"])
    num = 0
    for input_dataset in input_datasets:
        article = input_dataset.numpy().decode("utf-8")
        print(article)

        start_decoding = vocab.word_to_id(START_DECODING)
        stop_decoding = vocab.word_to_id(STOP_DECODING)

        article_words = article.split()[:params["max_enc_len"]]
        enc_input = [vocab.word_to_id(w) for w in article_words]
        enc_input_extend_vocab, article_oovs = article_to_ids(article_words, vocab)

        enc_hidden, enc_output = model.call_encoder(enc_input)
        dec_hidden = enc_hidden
        dec_input = tf.expand_dims(start_decoding, 0)

        result = ''
        for t in range(int(params['max_dec_len'])):
            predictions, _ = model(enc_output, enc_hidden, enc_input, enc_input_extend_vocab, dec_input, batch_oov_len=None)

            predicted_id = tf.argmax(predictions[0]).numpy()
            print('predicted_id', predicted_id)
            result += vocab.id_to_word(predicted_id) + ' '

            if vocab.id_to_word(predicted_id) == SENTENCE_END:
                print('Early stopping')
                break

            dec_input = tf.expand_dims([predicted_id], 1)
            print('dec_input:', dec_input)

        print('result: ', result)
        num += 1



if __name__ == '__main__':
    input_dataset = tf.data.TextLineDataset(params["test_seg_x_dir"])