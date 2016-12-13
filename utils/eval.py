from utils.data import load, dump
from models import MostFreqMatch, Embedding, GoldenRule
from collections import defaultdict as dd


class Evaluator(object):

    def __init__(self):
        self.test_set = load("mimic_episodes_index_test.pkl")
        self.golden_rule = load("icd_to_ndc_index.pkl")
        diag_vocab = load("diag_vocab.pkl")
        drug_vocab = load("drug_vocab.pkl")
        self.index_to_diag = {}
        self.index_to_drug = {}
        for diag in diag_vocab:
            self.index_to_diag[diag_vocab[diag]] = diag
        for drug in drug_vocab:
            self.index_to_drug[drug_vocab[drug]] = drug

    def eval(self, model):
        for pair in self.test_set:
            outputs = set(pair[1])
            prediction = set(model.predict(pair[0]))
            precision, recall = self.get_result(outputs, prediction)
            print(precision, recall)

    def eval_golden(self, model):
        for pair in self.test_set:
            all_output = []
            mapping = dd(list)
            for c1 in pair[0]:
                if c1 in self.golden_rule:
                    all_output.extend(self.golden_rule[c1])
                    for c2 in self.golden_rule[c1]:
                        mapping[c2].append(c1)
            all_output = set(all_output)
            prediction = set(model.predict(pair[0]))
            tp = prediction.intersection(all_output)

            precision = 0 if len(prediction) == 0 else float(len(tp)) / len(prediction)

            coverage = []
            for c in prediction:
                if c in mapping:
                    coverage.extend(mapping[c])
            tp = set(coverage).intersection(set(pair[0]))
            recall = 0 if len(pair[0]) == 0 else float(len(tp)) / len(pair[0])

            print([self.index_to_diag[c] for c in pair[0]])
            print([self.index_to_diag[c] for c in coverage])
            print([self.index_to_drug[c] for c in all_output])
            print([self.index_to_drug[c] for c in prediction])

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
    evaluator.eval_golden(mfm)


def eval_emb():
    evaluator = Evaluator()
    emb = Embedding()
    emb.load()
    evaluator.eval(emb)


def eval_golden():
    evaluator = Evaluator()
    golden = GoldenRule()
    evaluator.eval(golden)
    evaluator.eval_golden(golden)