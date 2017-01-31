from utils.data import load, dump
from collections import defaultdict as dd
from utils.eval import Evaluator

class MostFreqMatch(object):
    def __init__(self, k=3):
        self.freq = {}
        self.k = k

    def fit(self, train_set):
        freq = dd(lambda: dd(int))
        for pair in train_set:
            for t0 in pair[0]:
                for t1 in pair[1]:
                    freq[t0][t1] += 1
        for tk in freq:
            sorted_freq = sorted(freq[tk].items(), key=lambda x: x[1], reverse=True)
            self.freq[tk] = sorted_freq

        dump(dict(self.freq), "sutter_freq.pkl")

    def load(self, path="mimic_freq.pkl"):
        self.freq = load(path)

    def predict(self, inputs):
        outputs = []
        for token in inputs:
            if token in self.freq:
                for tk in self.freq[token][:self.k]:
                    outputs.append(tk[0])
        return list(set(outputs))


def train():
    input_vocab = load("sutter_diag_vocab.pkl")
    output_vocab = load("sutter_drug_vocab_3.pkl")
    train_set = load("sutter_encounter.train.pkl")
    test_set = load("sutter_encounter.dev.pkl")
    mfm = MostFreqMatch(1)
    mfm.fit(train_set)
    eva = Evaluator()

    sum_jaccard = 0
    cnt = 0
    for item in test_set:
        result = mfm.predict(item[0])
        sum_jaccard += eva.get_jaccard_k(item[1], result)


def eval_freq():
    mfm = MostFreqMatch()
    mfm.load()
    evaluator = Evaluator()
    evaluator.eval(mfm)
    evaluator.eval_golden(mfm)