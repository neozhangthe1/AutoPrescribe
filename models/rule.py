from utils.data import load, dump
from collections import defaultdict as dd
from utils.eval import Evaluator
import random

class RuleBased(object):
    def __init__(self, k=1):
        self.freq = {}
        self.k = k
        self.rule = None

    def load(self):
        self.rule = load("icd_gpi_map.pkl")[0]

    def predict(self, inputs):
        outputs = []
        for token in inputs:
            token = token.replace(".", "")
            if token in self.rule:
                rs = self.rule[token]
                if "" in rs:
                    rs.remove("")
                # for tk in list(self.rule[token])[:self.k]:
                outputs.append(random.choice(list(self.rule[token])))
        return list(set(outputs))


def eval_sutter():
    input_vocab = load("sutter_diag_vocab.pkl")
    output_vocab = load("sutter_drug_vocab_3.pkl")
    test_set = load("sutter_encounter.dev.pkl")
    rb = RuleBased(1)
    rb.load()
    results = []
    prediction_list = []
    truth_list = []
    for item in test_set:
        prediction = rb.predict(item[0])
        prediction_list.append(prediction)
        truth_list.append(item[1])
        results.append((item[0], item[1], prediction))
    dump(results, "sutter_result_rule.pkl")