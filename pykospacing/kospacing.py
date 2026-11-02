# -*- coding: utf-8 -*-
import os

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import re
import csv
from functools import lru_cache
from importlib import resources

import numpy as np

from pykospacing.embedding_maker import encoding_and_padding, load_vocab

__all__ = ['Spacing', ]

MAX_LEN = 198
BACKENDS = ('v1', 'v2')  # v1: KoSpacing v1 (TensorFlow), v2: KoSpacing v2 (tag model)


@lru_cache(maxsize=None)
def _tf_model():
    """The TensorFlow model and vocabulary, loaded on first use so that KoSpacing v2 needs no TensorFlow."""
    from tensorflow.keras.layers import TFSMLayer
    base = resources.files('pykospacing') / 'resources'
    model = TFSMLayer(str(base / 'models'), call_endpoint="serving_default")
    w2idx, _ = load_vocab(str(base / 'dicts' / 'c2v.dic'))
    return model, w2idx


class Spacing:
    """predict spacing for input string
    """

    def __init__(self, rules=[], backend='v1', checkpoint=None, device=None):
        """backend='v1' is KoSpacing v1, the original TensorFlow model. backend='v2' is KoSpacing v2, the
        instruction-following tag model (pykospacing/v2.py): checkpoint is its directory (or PYKOSPACING_V2_CHECKPOINT),
        device 'cuda' or 'cpu'."""
        if backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}")
        self.backend = backend
        if backend == 'v1':
            self._model, self._w2idx = _tf_model()
            self._v2 = None
        else:
            from pykospacing.v2 import KoSpacingV2
            self._v2 = KoSpacingV2(checkpoint, device)
        self.max_len = MAX_LEN
        self.pattern = re.compile(r'\s+')
        self.rules = {}
        for r in rules:
            if type(r) == str:
                self.rules[r] = re.compile(r'\s*'.join(r))
            else:
                raise ValueError("rules must to have only string values.")

    def set_rules_by_csv(self, file_path, key=None):
        with open(file_path, 'r', encoding='UTF-8') as csvfile:
            csv_var = csv.reader(csvfile)
            if key == None:
                for line in csv_var:
                    for word in line:
                        self.rules[word] = re.compile(r'\s*'.join(word))
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
                    self.rules[line[index]] = re.compile(r'\s*'.join(line[index]))

    def _require_v2(self, name):
        if self._v2 is None:
            raise NotImplementedError(f"{name}() needs KoSpacing v2: Spacing(backend='v2')")

    def fix(self, sent):
        """Correct spacing while keeping the spaces already in `sent` that are right (KoSpacing v2).
        A list of strings is processed as one batch and returns a list."""
        self._require_v2('fix')
        many = not isinstance(sent, str)
        out = [self.apply_rules(s) if self.rules else s for s in self._v2.fix(list(sent) if many else [sent])]
        return out if many else out[0]

    def space_probs(self, sent):
        """[(character, probability that a space follows)] over the non-space characters of `sent` (KoSpacing v2)."""
        self._require_v2('space_probs')
        many = not isinstance(sent, str)
        out = self._v2.space_probs(list(sent) if many else [sent])
        return out if many else out[0]

    def get_spaced_sent(self, raw_sent, deleted_str_list=None, deleted_idx_list=None, orig_sent=None, post_process=False):
        raw_sent_ = "«" + raw_sent + "»"
        raw_sent_ = raw_sent_.replace(' ', '^')
        sents_in = [raw_sent_, ]
        mat_in = encoding_and_padding(
            word2idx_dic=self._w2idx, sequences=sents_in, maxlen=200,
            padding='post', truncating='post')
        results = self._model(mat_in)
        mat_set = results['output_0'][0]
        preds = np.array(['1' if i > 0.5 else '0' for i in mat_set[:len(raw_sent_)]])
        if orig_sent is not None:
            orig_sent_ = "«" + orig_sent + "»"
            orig_sent_ = orig_sent_.replace(' ', '^')
            sents_in = [orig_sent_, ]
            mat_in = encoding_and_padding(
                word2idx_dic=self._w2idx, sequences=sents_in, maxlen=200,
                padding='post', truncating='post')
            results = self._model(mat_in)
            mat_set = results['output_0'][0]
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
        if self._v2 is not None:
            # the tag model reads Latin letters, digits and symbols directly, so `ignore` has nothing to do here
            if ignore != 'none':
                raise ValueError("ignore is only for KoSpacing v1 (backend='v1')")
            many = not isinstance(sent, str)
            out = self._v2.space(list(sent) if many else [sent])
            out = [self.apply_rules(s) if self.rules else s for s in out]
            return out if many else out[0]
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
