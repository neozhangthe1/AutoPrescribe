from utils.data import load, dump
from collections import defaultdict as dd


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

        dump(dict(self.freq), "mimic_freq.pkl")

    def load(self, path="mimic_freq.pkl"):
        self.freq = load(path)

    def predict(self, inputs):
        outputs = []
        for token in inputs:
            if token in self.freq:
                for tk in self.freq[token][:self.k]:
                    outputs.append(tk[0])
        return list(set(outputs))


