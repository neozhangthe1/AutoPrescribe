from utils.data import load, dump
from models import MostFreqMatch, Embedding


class Evaluator(object):

    def __init__(self):
        self.test_set = load("mimic_episodes_index_test.pkl")

    def eval(self, model):
        for pair in self.test_set:
            outputs = set(pair[1])
            prediction = set(model.predict(pair[0]))
            tp = len(outputs.intersection(prediction))
            fp = len(prediction - outputs)
            fn = len(outputs - prediction)
            precision = 0 if (tp + fp) == 0 else float(tp / (tp + fp))
            recall = 0 if (tp + fn) == 0 else float(tp / (tp + fn))
            print(precision, recall)


def eval_freq():
    evaluator = Evaluator()
    mfm = MostFreqMatch()
    mfm.load()
    evaluator.eval(mfm)


def eval_emb():
    evaluator = Evaluator()
    emb = Embedding()
    emb.load()
    evaluator.eval(emb)