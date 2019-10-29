import tensorflow as tf
from seq2seq_tf2.seq2seq_model import PGN
from seq2seq_tf2.batcher import Vocab, START_DECODING, STOP_DECODING, article_to_ids, output_to_words, SENTENCE_END
from seq2seq_tf2.preprocess import preprocess_sentence
from seq2seq_tf2 import config
import json

params = {'max_enc_len': 400,
          'max_dec_len': 100,
          'batch_size': 1,
          'vocab_size': 50000,
          'embed_size': 256,
          'enc_units': 256,
          'dec_units': 256,
          'attn_units': 512,
          'learning_rate': 0.015,
          'adagrad_init_acc': 0.1,
          'max_grad_norm': 0.8,
          'checkpoints_save_steps': 10,
          'mode': 'test',
          'pointer_gen': False,
          'model_dir': './ckpt',
          'train_seg_x_dir': '/Users/zn-nlp/Documents/project1_auto_master_qa/datasets/train_set.seg_x.txt',
          'train_seg_y_dir': '/Users/zn-nlp/Documents/project1_auto_master_qa/datasets/train_set.seg_y.txt',
          'test_seg_x_dir': '/Users/zn-nlp/Documents/project1_auto_master_qa/datasets/test_set.seg_x.txt',
          'vocab_path': '/Users/zn-nlp/Documents/project1_auto_master_qa/datasets/vocab.txt',
          'log_file': ''}


def test(sentence):
    vocab = Vocab(params["vocab_path"], params["vocab_size"])
    model = PGN(params)

    ckpt = tf.train.Checkpoint(model=model)
    checkpoint_dir = "{}/checkpoint".format(params["model_dir"])
    latest_ckpt = tf.train.latest_checkpoint(checkpoint_dir)
    ckpt.restore(latest_ckpt).expect_partial()

    sentence = preprocess_sentence(sentence)
    print('sentence is ', sentence)
    sentence_words = sentence.split()[:params["max_enc_len"]]
    print('sentence_words is ', sentence_words)
    enc_input = [vocab.word_to_id(w) for w in sentence_words]
    print('enc_input is ', enc_input)
    enc_input_extend_vocab, article_oovs = article_to_ids(sentence_words, vocab)
    print('enc_input_extend_vocab is ', enc_input_extend_vocab)
    print('article_oovs', article_oovs)

    start_decoding = vocab.word_to_id(START_DECODING)
    stop_decoding = vocab.word_to_id(STOP_DECODING)

    enc_input = tf.keras.preprocessing.sequence.pad_sequences([enc_input],
                                                           maxlen=params["max_enc_len"],
                                                           padding='post')
    print('enc_input is ', enc_input)
    enc_input = tf.convert_to_tensor(enc_input)
    print('enc_input is ', enc_input)

    enc_hidden, enc_output = model.call_encoder(enc_input)
    print('enc_hidden is ', enc_hidden)
    print('enc_output is ', enc_output)
    dec_hidden = enc_hidden
    dec_input = tf.expand_dims(start_decoding, 0)
    print('dec_input is ', dec_input)

    result = ''
    for t in range(int(params['max_dec_len'])):
        predictions, _ = model(enc_output, enc_hidden, enc_input, enc_input_extend_vocab, dec_input, batch_oov_len=None)
        print('predictions is ', predictions)

        predicted_id = tf.argmax(predictions[0]).numpy()
        print('predicted_id', predicted_id)
        result += vocab.id_to_word(predicted_id) + ' '

        if vocab.id_to_word(predicted_id) == SENTENCE_END:
            print('Early stopping')
            break

        dec_input = tf.expand_dims([predicted_id], 1)
        print('dec_input:', dec_input)

    print('result: ', result)


if __name__ == '__main__':
    test('我的帕萨特烧机油怎么办')

