from utils.data import load, dump
from collections import defaultdict as dd
import numpy as np

from models import MostFreqMatch, Embedding, GoldenRule


class Evaluator(object):

    def __init__(self, ds="mimic"):
        if ds == "mimic":
            self.test_set = load("mimic_episodes_index_test.pkl")
            self.golden_rule = load("icd_to_ndc_index.pkl")
            diag_vocab = load("diag_vocab.pkl")
            drug_vocab = load("drug_vocab.pkl")
        elif ds == "sutter":
            self.test_set = load("sutter_encounters_4.pkl")
            self.golden_rule = load("icd_to_ndc_index.pkl")
            diag_vocab = load("sutter_diag_vocab.pkl")
            drug_vocab = load("sutter_drug_vocab_4.pkl")
        self.index_to_diag = {}
        self.index_to_drug = {}
        for diag in diag_vocab:
            self.index_to_diag[diag_vocab[diag]] = diag
        for drug in drug_vocab:
            self.index_to_drug[drug_vocab[drug]] = drug

    def eval(self, model):
        precisions, recalls = [], []
        cnt = 0
        for pair in self.test_set:
            outputs = set(pair[1])
            prediction = set(model.predict(pair[0]))
            precision, recall = self.get_result(outputs, prediction)
            precisions.append(precision)
            recalls.append(recall)
            if cnt % 1000 == 0:
                print(cnt, precision, recall)
                print(np.mean(precisions), np.mean(recalls))
            cnt += 1
        print(np.mean(precisions), np.mean(recalls))

    def eval_golden(self, model):
        precisions, recalls = [], []
        cnt = 0
        for pair in self.test_set:
            prediction = model.predict(pair[0])
            precision, recall = self.get_golden_eval(pair[0], prediction)
            precisions.append(precision)
            recalls.append(recall)
            if cnt % 1000 == 0:
                print(cnt, precision, recall)
                print(np.mean(precisions), np.mean(recalls))
            cnt += 1
        print(np.mean(precisions), np.mean(recalls))

    @staticmethod
    def get_result(truth, prediction):
        truth = set(truth)
        prediction = set(prediction)
        tp = len(truth.intersection(prediction))
        fp = len(prediction - truth)
        fn = len(truth - prediction)
        precision = 0 if (tp + fp) == 0 else float(tp / (tp + fp))
        recall = 0 if (tp + fn) == 0 else float(tp / (tp + fn))

        return precision, recall
    
    def get_golden_eval(self, inputs, prediction):
        prediction = set(prediction)
        all_output = []
        mapping = dd(list)
        for c1 in inputs:
            if c1 in self.golden_rule:
                all_output.extend(self.golden_rule[c1])
                for c2 in self.golden_rule[c1]:
                    mapping[c2].append(c1)
        all_output = set(all_output)
        tp = prediction.intersection(all_output)

        precision = 0 if len(prediction) == 0 else float(len(tp)) / len(prediction)

        coverage = []
        for c in prediction:
            if c in mapping:
                coverage.extend(mapping[c])
        tp = set(coverage).intersection(set(inputs))
        recall = 0 if len(inputs) == 0 else float(len(tp)) / len(inputs)

        return precision, recall


def eval_freq():
    mfm = MostFreqMatch()
    mfm.load()
    evaluator = Evaluator()
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


def eval_real():
    test_set = load("mimic_episodes_index_test.pkl")
    evaluator = Evaluator()
    precisions, recalls = [], []
    cnt = 0
    for pair in test_set:
        precision, recall = evaluator.get_golden_eval(pair[0], pair[1])
        precisions.append(precision)
        recalls.append(recall)
        if cnt % 1000 == 0:
            print(cnt, precision, recall)
            print(np.mean(precisions), np.mean(recalls))
        cnt += 1


