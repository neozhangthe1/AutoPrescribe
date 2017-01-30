import pickle
import random
from collections import defaultdict as dd

import numpy as np


class structure:
    def __init__(self, source_text, target_text):
        self.source_text, self.target_text = source_text, target_text


class Processor:
    def __init__(self, config):
        self.train_data = self.load(config.train_pkl)
        self.dev_data = self.load(config.dev_pkl)
        self.source_vocab = self.load(config.source_vocab_pkl)
        self.target_vocab = self.load(config.target_vocab_pkl)

        self.config = config
        self.idx_to_source_token, self.idx_to_target_token = self.build_index()

        self.source_vocab_size = len(self.source_vocab)
        self.target_vocab_size = len(self.target_vocab)
        print('train_size', len(self.train_data))
        print('dev_size', len(self.dev_data))
        print('source vocab size', self.source_vocab_size)
        print('target vocab size', self.target_vocab_size)

        random.seed(config.model_seed)

    def get_char_index(self, token, source=True):
        if source:
            if token in self.source_vocab:
                return self.source_vocab[token]
            return self.source_vocab['UNK']
        else:
            if token in self.target_vocab:
                return self.target_vocab[token]
            return self.target_vocab['UNK']

    def decode(self, sample_y, ref):
        END_INDEX = self.target_vocab['END']
        s = []
        for i in range(sample_y.shape[0]):
            char_index = sample_y[i]
            if char_index == END_INDEX: break
            s.append(self.idx_to_target_token[char_index])
        return s

    def gen_batch(self, data_, batch_size=None, shuffle=True):
        if batch_size is None:
            batch_size = self.config.batch_size

        data = data_

        config = self.config
        if shuffle:
            random.shuffle(data)

        def get_data_structure():
            source_inputs = np.zeros((batch_size, config.source_len), dtype=np.int32)
            target_inputs = np.zeros((batch_size, config.target_len), dtype=np.int32)
            target_outputs = np.zeros((batch_size, config.target_len), dtype=np.int32)
            source_mask_inputs = np.zeros((batch_size, config.source_len), dtype=np.float32)
            target_mask_inputs = np.zeros((batch_size, config.target_len), dtype=np.float32)
            refs = []

            return source_inputs, target_inputs, target_outputs, source_mask_inputs, target_mask_inputs, refs

        source_inputs, target_inputs, target_outputs, source_mask_inputs, target_mask_inputs, refs = get_data_structure()

        i = 0
        # UNK_INDEX = self.get_char_index('UNK')
        for pair in data:
            target_text, source_text = tuple(pair)

            max_target_len = min(len(target_text) + 1, config.target_len)
            target_inputs[i, 0] = self.get_char_index('START', False)
            if max_target_len == len(target_text) + 1:
                target_outputs[i, max_target_len - 1] = self.get_char_index('END', False)
            else:
                target_outputs[i, max_target_len - 1] = self.get_char_index(target_text[max_target_len - 1])
            for j in range(max_target_len - 1):
                target_inputs[i, j + 1] = self.get_char_index(target_text[j], False)
                target_outputs[i, j] = self.get_char_index(target_text[j], False)
            target_mask_inputs[i, : max_target_len] = 1.0

            for j in range(len(source_text)):
                if j >= config.source_len: break
                source_inputs[i, j] = self.get_char_index(source_text[j])
            source_mask_inputs[i, : min(len(source_text), config.source_len)] = 1.0

            # extra2index, extra_cnt, extras = {}, 0, []
            # for j in range(len(source_text)):
            #     if j >= config.source_len: break
            #     char_index = self.get_char_index(source_text[j])
            #     char = source_text[j]
            #     if char_index == UNK_INDEX:
            #         if char not in extra2index:
            #             extra2index[char] = extra_cnt
            #             extra_cnt += 1
            #             extras.append(char)
            #         extra_index = extra2index[char]
            #         map_inputs[i, j, self.char_cnt + extra_index] = 1.0
            #     else:
            #         map_inputs[i, j, char_index] = 1.0
            # for j in range(max_target_len - 1):
            #     if target_outputs[i, j] == UNK_INDEX and j < len(target_text):
            #         char = target_text[j]
            #         if char in extra2index:
            #             target_outputs[i, j] = self.char_cnt + extra2index[char]
            refs.append(structure(source_text, target_text))

            i += 1
            if i == batch_size:
                yield source_inputs, target_inputs, target_outputs, source_mask_inputs, target_mask_inputs, refs
                source_inputs, target_inputs, target_outputs, source_mask_inputs, target_mask_inputs, refs = get_data_structure()
                i = 0
        if i > 0:
            yield source_inputs[:i], target_inputs[:i], target_outputs[:i], source_mask_inputs[:i], target_mask_inputs[
                                                                                                    :i], refs[:i]

    def gen_one_batch(self, refs):
        config = self.config
        batch_size = len(refs)

        source_inputs = np.zeros((batch_size, config.source_len), dtype=np.int32)
        target_inputs = np.zeros((batch_size, config.target_len), dtype=np.int32)
        target_outputs = np.zeros((batch_size, config.target_len), dtype=np.int32)
        source_mask_inputs = np.zeros((batch_size, config.source_len), dtype=np.float32)
        target_mask_inputs = np.zeros((batch_size, config.target_len), dtype=np.float32)
        map_inputs = np.zeros((batch_size, config.source_len, self.char_cnt + self.extra_char_cnt), dtype=np.float32)

        UNK_INDEX = self.get_char_index('UNK')
        for i, ref in enumerate(refs):
            target_text, source_text = ref.target_text, ref.source_text

            max_target_len = min(len(target_text) + 1, config.target_len)
            target_inputs[i, 0] = self.get_char_index('START')
            if max_target_len == len(target_text) + 1:
                target_outputs[i, max_target_len - 1] = self.get_char_index('END')
            else:
                target_outputs[i, max_target_len - 1] = self.get_char_index(target_text[max_target_len - 1])
            for j in range(max_target_len - 1):
                target_inputs[i, j + 1] = self.get_char_index(target_text[j])
                target_outputs[i, j] = self.get_char_index(target_text[j])
            target_mask_inputs[i, : max_target_len] = 1.0

            for j in range(len(source_text)):
                if j >= config.source_len: break
                source_inputs[i, j] = self.get_char_index(source_text[j])
            source_mask_inputs[i, : min(len(source_text), config.source_len)] = 1.0

            extra2index, extra_cnt, extras = {}, 0, []
            for j in range(len(source_text)):
                if j >= config.source_len: break
                char_index = self.get_char_index(source_text[j])
                char = source_text[j]
                if char_index == UNK_INDEX:
                    if char not in extra2index:
                        extra2index[char] = extra_cnt
                        extra_cnt += 1
                        extras.append(char)
                    extra_index = extra2index[char]
                    map_inputs[i, j, self.char_cnt + extra_index] = 1.0
                else:
                    map_inputs[i, j, char_index] = 1.0
            for j in range(max_target_len - 1):
                if target_outputs[i, j] == UNK_INDEX and j < len(target_text):
                    char = target_text[j]
                    if char in extra2index:
                        target_outputs[i, j] = self.char_cnt + extra2index[char]

        return source_inputs, target_inputs, target_outputs, source_mask_inputs, target_mask_inputs, map_inputs

    def build_index(self):
        self.target_vocab["START"] = max(self.target_vocab.items()) + 1
        self.target_vocab["END"] = max(self.target_vocab.items()) + 1
        self.target_vocab["UNK"] = max(self.target_vocab.items()) + 1
        idx_to_source_token = dd(int)
        idx_to_target_token = dd(int)
        for token in self.source_vocab:
            idx_to_source_token[self.source_vocab[token]] = token
        for token in self.target_vocab:
            idx_to_target_token[self.target_vocab[token]] = token

        return idx_to_source_token, idx_to_target_token

    def load(self, filename):
        return pickle.load(open(filename, "rb"))
