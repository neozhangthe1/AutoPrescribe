import pickle
import os
import csv
from collections import defaultdict as dd
import codecs
import random
import numpy as np

directory = "data/"
model_directory = "build/"

_PAD = b"_PAD"
_GO = b"_GO"
_EOS = b"_EOS"
_UNK = b"_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 4127
GO_ID = 4128
EOS_ID = 4129
UNK_ID = 4130



class structure:
    def __init__(self, source_text, target_text, extras):
        self.source_text, self.target_text, self.extras = source_text, target_text, extras

class Processor:
    def __init__(self, config):
        self.train_data = self.load(config.train_pkl)
        self.dev_data = self.load(config.dev_pkl)
        self.source_vocab_size = None
        self.target_vocab_size = None
        print('train_size', len(self.train_data))
        print('dev_size', len(self.dev_data))
        self.config = config
        self.build_index()

        self.extra_char_cnt = config.source_len

        random.seed(config.model_seed)

    def get_char_index(self, char):
        if char in self.char2index:
            return self.char2index[char]
        return self.char2index['UNK']

    def decode(self, sample_y, ref):
        END_INDEX = self.char2index['END']
        s = []
        for i in range(sample_y.shape[0]):
            char_index = sample_y[i]
            if char_index == END_INDEX: break
            if char_index < self.char_cnt:
                s.append(self.index2char[char_index])
            else:
                s.append(ref.extras[char_index - self.char_cnt])
        return s

    def gen_batch(self, data_, batch_size = None, shuffle = True):
        if batch_size is None:
            batch_size = self.config.batch_size

        data = data_

        config = self.config
        if shuffle:
            random.shuffle(data)

        def get_data_structure():
            source_inputs = np.zeros((batch_size, config.source_len), dtype = np.int32)
            target_inputs = np.zeros((batch_size, config.target_len), dtype = np.int32)
            target_outputs = np.zeros((batch_size, config.target_len), dtype = np.int32)
            source_mask_inputs = np.zeros((batch_size, config.source_len), dtype = np.float32)
            target_mask_inputs = np.zeros((batch_size, config.target_len), dtype = np.float32)
            map_inputs = np.zeros((batch_size, config.source_len, self.char_cnt + self.extra_char_cnt), dtype = np.float32)
            refs = []

            return source_inputs, target_inputs, target_outputs, source_mask_inputs, target_mask_inputs, map_inputs, refs

        source_inputs, target_inputs, target_outputs, source_mask_inputs, target_mask_inputs, map_inputs, refs = get_data_structure()

        i = 0
        UNK_INDEX = self.get_char_index('UNK')
        for pair in data:
            target_text, source_text = tuple(pair)

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
            refs.append(structure(source_text, target_text, extras))

            i += 1
            if i == batch_size:
                yield source_inputs, target_inputs, target_outputs, source_mask_inputs, target_mask_inputs, map_inputs, refs
                source_inputs, target_inputs, target_outputs, source_mask_inputs, target_mask_inputs, map_inputs, refs = get_data_structure()
                i = 0
        if i > 0:
            yield source_inputs[:i], target_inputs[:i], target_outputs[:i], source_mask_inputs[:i], target_mask_inputs[:i], map_inputs[:i], refs[:i]

    def gen_one_batch(self, refs):
        config = self.config
        batch_size = len(refs)

        source_inputs = np.zeros((batch_size, config.source_len), dtype = np.int32)
        target_inputs = np.zeros((batch_size, config.target_len), dtype = np.int32)
        target_outputs = np.zeros((batch_size, config.target_len), dtype = np.int32)
        source_mask_inputs = np.zeros((batch_size, config.source_len), dtype = np.float32)
        target_mask_inputs = np.zeros((batch_size, config.target_len), dtype = np.float32)
        map_inputs = np.zeros((batch_size, config.source_len, self.char_cnt + self.extra_char_cnt), dtype = np.float32)

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
        char2cnt = dd(int)
        for item in self.train_data:
            for s in item:
                for char in s:
                    char2cnt[char] += 1
        char2cnt = [(k, v) for k, v in char2cnt.iteritems()]
        char2cnt.sort(key = lambda x: x[1], reverse = True)
        char2cnt = char2cnt[: self.config.vocab_size]

        char2index, index2char = {}, ['PAD', 'UNK', 'START', 'END']
        for k, _ in char2cnt:
            index2char.append(k)
        for i, char in enumerate(index2char):
            char2index[char] = i
        self.char2index, self.index2char, self.char_cnt = char2index, index2char, len(index2char)
        print('char_cnt', self.char_cnt)

    def load(self, filename):
        return pickle.load(open(filename))


def get_path(path):
    return os.path.join(directory, path)


def get_model_path(path):
    return os.path.join(model_directory, path)


def clean_data():
    train_set = load("mimic_episodes_train.pkl")
    test_set = load("mimic_episodes_test.pkl")

    def clean(src_set):
        for pair in src_set:
            if "0" in pair[1]:
                pair[1].remove("0")
            if '' in pair[1]:
                pair[1].remove('')

    dump(train_set, "mimic_episodes_train.pkl")
    dump(test_set, "mimic_episodes_test.pkl")


def to_index(src_set, input_vocab, output_vocab):
    target_set = []
    for pair in src_set:
        inputs = []
        for token in pair[0]:
            if token in input_vocab:
                inputs.append(input_vocab[token])
        outputs = []
        for token in pair[1]:
            if token in output_vocab:
                outputs.append(output_vocab[token])
        target_set.append([inputs, outputs])
    return target_set


def run_to_index():
    train_set = pickle.load(open(get_path("mimic_episodes_train.pkl"), "rb"))
    test_set = pickle.load(open(get_path("mimic_episodes_test.pkl"), "rb"))
    input_vocab = pickle.load(open(get_path("diag_vocab.pkl"), "rb"))
    output_vocab = pickle.load(open(get_path("drug_vocab.pkl"), "rb"))
    train_index_set = to_index(train_set, input_vocab, output_vocab)
    test_index_set = to_index(test_set, input_vocab, output_vocab)
    with open(get_path("mimic_episodes_index_train.pkl"), "wb") as f_out:
        pickle.dump(train_index_set, f_out)
    with open(get_path("mimic_episodes_index_test.pkl"), "wb") as f_out:
        pickle.dump(test_index_set, f_out)


def downgrade_pickle():
    for f in ["mimic_episodes_test.pkl", "mimic_episodes_train.pkl", "diag_vocab.pkl", "drug_vocab.pkl", "mimic_episodes.pkl"]:
        with open(f, "rb") as f_in:
            d = pickle.load(f_in)
            with open(f+'1', "wb") as f_out:
                pickle.dump(d, f_out, protocol=2)


def load_rx_gpi_mapping():
    with open(get_path("ndw_v_product.txt")) as f_in:
        titles = next(f_in).strip().split("|")
        for line in f_in:
            pass


def dump(obj, path):
    with open(get_path(path), "wb") as f_out:
        pickle.dump(obj, f_out)


def load(path):
    return pickle.load(open(get_path(path), "rb"))


if __name__ == "__main__":
    run_to_index()

