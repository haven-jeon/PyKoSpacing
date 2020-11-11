# -*- coding: utf-8 -*-
import os
import re

import numpy as np
import pkg_resources
from tensorflow.keras.models import load_model
from pykospacing.embedding_maker import encoding_and_padding, load_vocab

__all__ = ['spacing', ]
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

model_path = pkg_resources.resource_filename(
    'pykospacing', os.path.join('resources', 'models', 'kospacing'))
dic_path = pkg_resources.resource_filename(
    'pykospacing', os.path.join('resources', 'dicts', 'c2v.dic'))
MODEL = load_model(model_path)
MODEL.make_predict_function()
W2IDX, _ = load_vocab(dic_path)


class PredSpacing:
    """predict spacing for input string
    """
    def __init__(self, model, w2idx):
        self._model = model
        self._w2idx = w2idx
        self.pattern = re.compile(r'\s+')

    def get_spaced_sent(self, raw_sent):
        raw_sent_ = "«" + raw_sent + "»"
        raw_sent_ = raw_sent_.replace(' ', '^')
        sents_in = [raw_sent_, ]
        mat_in = encoding_and_padding(
            word2idx_dic=self._w2idx, sequences=sents_in, maxlen=200,
            padding='post', truncating='post')
        results = self._model.predict(mat_in)
        mat_set = results[0, ]
        preds = np.array(
            ['1' if i > 0.5 else '0' for i in mat_set[:len(raw_sent_)]])
        return self.make_pred_sents(raw_sent_, preds)

    def make_pred_sents(self, x_sents, y_pred):
        res_sent = []
        for i, j in zip(x_sents, y_pred):
            if j == '1':
                res_sent.append(i)
                res_sent.append(' ')
            else:
                res_sent.append(i)
        subs = re.sub(self.pattern, ' ', ''.join(res_sent).replace('^', ' '))
        subs = subs.replace('«', '')
        subs = subs.replace('»', '')
        return subs


PredSpacing = PredSpacing(MODEL, W2IDX)

MAX_LEN = 198


def spacing(sent):
    if len(sent) > MAX_LEN:
        splitted_sent = [sent[y-MAX_LEN:y] for y in range(MAX_LEN, len(sent)+MAX_LEN, MAX_LEN)]
        spaced_sent = ''.join([PredSpacing.get_spaced_sent(ss)
                               for ss in splitted_sent])
    else:
        spaced_sent = PredSpacing.get_spaced_sent(sent)
    return spaced_sent.strip()
