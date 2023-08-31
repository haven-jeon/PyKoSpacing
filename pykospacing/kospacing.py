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

    def get_spaced_sent(self, raw_sent, deleted_str_list=None, deleted_idx_list=None, orig_sent=None, post_process=False):
        raw_sent_ = "«" + raw_sent + "»"
        raw_sent_ = raw_sent_.replace(' ', '^')
        sents_in = [raw_sent_, ]
        mat_in = encoding_and_padding(
            word2idx_dic=self._w2idx, sequences=sents_in, maxlen=200,
            padding='post', truncating='post')
        results = self._model.predict(mat_in, verbose=0)
        mat_set = results[0,]
        preds = np.array(['1' if i > 0.5 else '0' for i in mat_set[:len(raw_sent_)]])
        if orig_sent is not None:
            orig_sent_ = "«" + orig_sent + "»"
            orig_sent_ = orig_sent_.replace(' ', '^')
            sents_in = [orig_sent_, ]
            mat_in = encoding_and_padding(
                word2idx_dic=self._w2idx, sequences=sents_in, maxlen=200,
                padding='post', truncating='post')
            results = self._model.predict(mat_in, verbose=0)
            mat_set = results[0,]
            orig_preds = np.array(['1' if i > 0.5 else '0' for i in mat_set[:len(orig_sent_)]])
        else:
            orig_sent_ = None
            orig_preds = None
        return self.make_pred_sents(raw_sent_, preds, orig_sent_, orig_preds, deleted_str_list, deleted_idx_list, post_process)

    def make_pred_sents(self, x_sents, y_pred,
                        orig_sent_=None, orig_preds=None, deleted_str_list=None, deleted_idx_list=None, post_process=False):
        res_sent = []
        accum_idx = 0
        adjusted_idx = 0
        for i in range(len(x_sents)):
            res_sent.append(x_sents[i])
            if deleted_str_list:  # if ignore is 'pre' or 'pre2' and deleted_str is not empty
                if deleted_idx_list[accum_idx] == i:  # if have to insert deleted_str
                    if orig_preds is not None and orig_preds[adjusted_idx] == '1':
                        # if ignore is 'pre2' and have to insert space before deleted_str
                        res_sent.append(' ')
                    res_sent.append(deleted_str_list[accum_idx])  # insert deleted_str
                    adjusted_idx += len(deleted_str_list[accum_idx])
                    if accum_idx < len(deleted_idx_list) - 1:
                        accum_idx += 1
                    if orig_preds is not None and orig_preds[adjusted_idx] == '1':
                        # if ignore is 'pre2' and have to insert space after deleted_str
                        res_sent.append(' ')
                    if orig_preds is None and y_pred[i] == '1':
                        # if ignore is 'pre' and have to insert space, just insert space after deleted_str (it is not perfect)
                        res_sent.append(' ')
                else:
                    if y_pred[i] == '1':
                        if post_process and deleted_idx_list:
                            # if ignore is 'post' and have to ignore space, just insert space before the character
                            for start, end in deleted_idx_list:
                                if start <= i < end:
                                    break
                            else:
                                # only append space if the index is not in the range of deleted_idx_list
                                res_sent.append(' ')
                        else:
                            res_sent.append(' ')
            else:  # if ignore is 'none' or 'post' or deleted_str is empty
                if y_pred[i] == '1':
                    res_sent.append(' ')
            adjusted_idx += 1
        subs = re.sub(self.pattern, ' ', ''.join(res_sent).replace('^', ' '))
        subs = subs.replace('«', '')
        subs = subs.replace('»', '')
        return subs

    def apply_rules(self, spaced_sent):
        for word, rgx in self.rules.items():
            spaced_sent = rgx.sub(word, spaced_sent)
        return spaced_sent

    def __call__(self, sent, ignore='none',
                 ignore_pattern=r'[^가-힣ㄱ-ㅣ!-@[-`{-~\s]+,*( [^가-힣ㄱ-ㅣ!-@[-`{-~\s]+,*)*[.,!?]* *'):
        assert ignore in ['none', 'pre', 'pre2', 'post']
        if len(sent) > self.max_len:
            splitted_sent = [sent[y - self.max_len:y] for y in
                             range(self.max_len, len(sent) + self.max_len, self.max_len)]
        else:
            splitted_sent = [sent]
        result_sent = []
        for i in range(len(splitted_sent)):
            if ignore in ['pre', 'pre2']:
                # delete the characters that are not Korean or symbol *first*.
                # also save the deleted characters and their indices.
                # the format of deleted_idx_list is different from the below.
                # (it is a list of integers, index of filtered_sent)
                filtered_sent = re.sub(ignore_pattern, '', splitted_sent[i])
                deleted_str_list = []
                deleted_idx_list = []
                accum_len = 0
                for deleted_str in re.finditer(ignore_pattern, splitted_sent[i]):
                    deleted_str_list.append(deleted_str.group())
                    deleted_idx_list.append(deleted_str.span()[0] - accum_len)
                    accum_len += len(deleted_str.group())
            elif ignore == 'post':
                # just save the index of the characters that are not Korean or symbol.
                # the characters will be deleted *later*.
                # the format of deleted_idx_list is different from the above.
                # (it is a list of tuples, index of splitted_sent)
                filtered_sent = splitted_sent[i]
                deleted_str_list = None
                deleted_idx_list = []
                for deleted_str in re.finditer(ignore_pattern, splitted_sent[i]):
                    deleted_idx_list.append((deleted_str.span()[0], deleted_str.span()[1]))
            else:
                # if ignore == 'none', do nothing.
                filtered_sent = splitted_sent[i]
                deleted_str_list = None
                deleted_idx_list = None
            # if ignore == 'pre2', orig_sent is needed for double prediction
            orig_sent = splitted_sent[i] if ignore == 'pre2' else None
            # if ignore == 'post', set post_process to True
            post_process = True if ignore == 'post' else False
            spaced_sent = self.get_spaced_sent(filtered_sent, deleted_str_list, deleted_idx_list, orig_sent, post_process)
            result_sent.append(spaced_sent)
        spaced_sent = ''.join(result_sent)
        if len(self.rules) > 0:
            spaced_sent = self.apply_rules(spaced_sent)
        return spaced_sent.strip()
