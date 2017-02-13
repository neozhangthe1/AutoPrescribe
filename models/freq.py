from utils.data import load, dump
from collections import defaultdict as dd
from utils.eval import Evaluator


class MostFreqMatch(object):
    def __init__(self, k=1, data="sutter"):
        self.freq = {}
        self.k = k
        self.data = data

    def fit(self, train_set):
        freq = dd(lambda: dd(int))
        for pair in train_set:
            for t0 in pair[0]:
                for t1 in pair[1]:
                    freq[t0][t1] += 1
        for tk in freq:
            sorted_freq = sorted(freq[tk].items(), key=lambda x: x[1], reverse=True)
            self.freq[tk] = sorted_freq

        dump(dict(self.freq), self.data + "_freq.pkl")

    def load(self, path):
        if path is None:
            path = self.data + "_freq.pkl"
        self.freq = load(path)

    def predict(self, inputs):
        outputs = []
        for token in inputs:
            if token in self.freq:
                for tk in self.freq[token][:self.k]:
                    outputs.append(tk[0])
        return list(set(outputs))


def train():
    level = 2
    # input_vocab = load("sutter_diag_vocab.pkl")
    # output_vocab = load("sutter_drug_vocab_%s.pkl" % level)
    train_set = load("sutter_encounters_%s.train.pkl" % level)
    test_set = load("sutter_encounters_%s.test.pkl" % level)
    mfm = MostFreqMatch(1)
    mfm.fit(train_set)

    results = []
    prediction_list = []
    truth_list = []
    for item in test_set:
        prediction = mfm.predict(item[0])
        prediction_list.append(prediction)
        truth_list.append(item[1])
        results.append((item[0], item[1], prediction))
    dump(results, "sutter_result_freq_%s.pkl" % level)

    # eva = Evaluator()
    #
    # sum_jaccard = 0
    # cnt = 0
    # for item in test_set:
    #     result = mfm.predict(item[0])
    #     sum_jaccard += eva.get_jaccard_k(item[1], result)

def train_mimic():
    train_set = load("mimic_encounter_gpi.train.pkl")
    test_set = load("mimic_encounter_gpi.dev.pkl")
    mfm = MostFreqMatch(3, "mimic")
    mfm.fit(train_set)
    results = []
    prediction_list = []
    truth_list = []
    for item in test_set:
        prediction = mfm.predict(item[0])
        prediction_list.append(prediction)
        truth_list.append(item[1])
        results.append((item[0], item[1], prediction))
    dump(results, "mimic_result_freq.pkl")


def eval_freq():
    level = 2
    input_vocab = load("sutter_diag_vocab.pkl")
    output_vocab = load("sutter_drug_vocab_%s.pkl" % level)
    test_set = load("sutter_encounters.test_%s.pkl" % level)
    mfm = MostFreqMatch(1)
    mfm.load("sutter_freq.pkl")
    results = []
    prediction_list = []
    truth_list = []
    for item in test_set:
        prediction = mfm.predict(item[0])
        prediction_list.append(prediction)
        truth_list.append(item[1])
        results.append((item[0], item[1], prediction))
    dump(results, "sutter_result_freq_%s.pkl" % level)


    evaluator = Evaluator()
    evaluator.eval(mfm)
    evaluator.eval_golden(mfm)