# -*- coding: utf-8 -*-
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import re
import csv

import numpy as np
import pkg_resources
from tensorflow.keras.models import load_model
from pykospacing.embedding_maker import encoding_and_padding, load_vocab

__all__ = ['Spacing', ]


model_path = pkg_resources.resource_filename(
    'pykospacing', os.path.join('resources', 'models', 'kospacing'))
dic_path = pkg_resources.resource_filename(
    'pykospacing', os.path.join('resources', 'dicts', 'c2v.dic'))
MODEL = load_model(model_path)
MODEL.make_predict_function()
W2IDX, _ = load_vocab(dic_path)
MAX_LEN = 198

def is_korean_or_symbol(char):
    if ord('가') <= ord(char) <= ord('힣'):
        return True
    if ord('ㄱ') <= ord(char) <= ord('ㅣ'):
        return True
    if ord('!') <= ord(char) <= ord('@'):
        return True
    if ord('[') <= ord(char) <= ord('`'):
        return True
    if ord('{') <= ord(char) <= ord('~'):
        return True
    return False

class Spacing:
    """predict spacing for input string
    """
    def __init__(self, rules=[]):
        self._model = MODEL
        self._w2idx = W2IDX
        self.max_len = MAX_LEN
        self.pattern = re.compile(r'\s+')
        self.rules = {}
        for r in rules:
            if type(r) == str:
                self.rules[r] = re.compile('\s*'.join(r))
            else:
                raise ValueError("rules must to have only string values.")
    
    def set_rules_by_csv(self, file_path, key=None):
        with open(file_path, 'r', encoding='UTF-8') as csvfile:
            csv_var = csv.reader(csvfile)
            if key == None:
                for line in csv_var:
                    for word in line:
                        self.rules[word] = re.compile('\s*'.join(word))
            else:
                csv_var = list(csv_var)
                index = -1
                for i, word in enumerate(csv_var[0]):
                    if word == key:
                        index = i
                        break
                
                if index == -1:
                    raise KeyError(f"'{key}' is not in csv file")
                
                for line in csv_var:
                    self.rules[line[index]] = re.compile('\s*'.join(line[index]))

    def get_spaced_sent(self, raw_sent, only_ko=False):
        raw_sent_ = "«" + raw_sent + "»"
        raw_sent_ = raw_sent_.replace(' ', '^')
        sents_in = [raw_sent_, ]
        mat_in = encoding_and_padding(
            word2idx_dic=self._w2idx, sequences=sents_in, maxlen=200,
            padding='post', truncating='post')
        results = self._model.predict(mat_in, verbose=0)
        mat_set = results[0, ]
        preds = np.array(
            ['1' if i > 0.5 else '0' for i in mat_set[:len(raw_sent_)]])
        return self.make_pred_sents(raw_sent_, preds, only_ko)

    def make_pred_sents(self, x_sents, y_pred, only_ko=False):
        res_sent = []
        for char_idx, (i, j) in enumerate(zip(x_sents, y_pred)):
            if j == '1':
                res_sent.append(i)
                if only_ko:
                    if is_korean_or_symbol(i) or (char_idx < len(y_pred)-1 and is_korean_or_symbol(x_sents[char_idx+1])):
                        res_sent.append(' ')
                else:
                    res_sent.append(' ')
            else:
                res_sent.append(i)
        subs = re.sub(self.pattern, ' ', ''.join(res_sent).replace('^', ' '))
        subs = subs.replace('«', '')
        subs = subs.replace('»', '')
        return subs

    def apply_rules(self, spaced_sent):
        for word, rgx in self.rules.items():
            spaced_sent = rgx.sub(word, spaced_sent)
        return spaced_sent

    def __call__(self, sent, only_ko=False):
        if len(sent) > self.max_len:
            splitted_sent = [sent[y-self.max_len:y] for y in range(self.max_len, len(sent)+self.max_len, self.max_len)]
            spaced_sent = ''.join([self.get_spaced_sent(ss, only_ko)
                                for ss in splitted_sent])
        else:
            spaced_sent = self.get_spaced_sent(sent, only_ko)
        if len(self.rules) > 0:
            spaced_sent = self.apply_rules(spaced_sent)
        return spaced_sent.strip()
