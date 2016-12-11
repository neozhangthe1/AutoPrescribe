from utils.data import load, dump
from models import MostFreqMatch, Embedding


class Evaluator(object):

    def __init__(self):
        self.test_set = load("mimic_episodes_index_test.pkl")

    def eval(self, model):
        for pair in self.test_set:
            outputs = set(pair[1])
            prediction = set(model.predict(pair[0]))
            precision, recall = self.get_result(outputs, prediction)
            print(precision, recall)

    @staticmethod
    def get_result(truth, prediction):
        tp = len(truth.intersection(prediction))
        fp = len(prediction - truth)
        fn = len(truth - prediction)
        precision = 0 if (tp + fp) == 0 else float(tp / (tp + fp))
        recall = 0 if (tp + fn) == 0 else float(tp / (tp + fn))
        return precision, recall


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