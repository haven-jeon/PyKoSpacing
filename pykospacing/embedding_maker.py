from tensorflow.keras.preprocessing import sequence
import json
import numpy as np


__all__ = ['load_embedding', 'load_vocab', 'encoding_and_padding']


def load_embedding(embeddings_file):
    return(np.load(embeddings_file))


def load_vocab(vocab_path):
    with open(vocab_path, 'r') as f:
        data = json.loads(f.read())
    word2idx = data
    idx2word = dict([(v, k) for k, v in data.items()])
    return word2idx, idx2word


def encoding_and_padding(word2idx_dic, sequences, **params):
    """
    1. making item to idx
    2. padding

    :word2idx_dic
    :sequences: list of lists where each element is a sequence
    :maxlen: int, maximum length
    :dtype: type to cast the resulting sequence.
    :padding: 'pre' or 'post', pad either before or after each sequence.
    :truncating: 'pre' or 'post', remove values from sequences larger than
        maxlen either in the beginning or in the end of the sequence
    :value: float, value to pad the sequences to the desired value.
    """
    seq_idx = [[word2idx_dic.get(a, word2idx_dic['__ETC__']) for a in i] for i in sequences]
    params['value'] = word2idx_dic['__PAD__']
    return(sequence.pad_sequences(seq_idx, **params))
