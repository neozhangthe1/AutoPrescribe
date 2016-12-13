from utils.data import load, dump
import random


class GoldenRule(object):
    def __init__(self):
        self.icd_to_ndc = load("icd_to_ndc_index.pkl")
        self.k = 3

    def predict(self, inputs):
        outputs = []
        for token in inputs:
            if token in self.icd_to_ndc:
                if len(self.icd_to_ndc[token]) > 0:
                    outputs.extend(random.sample(self.icd_to_ndc[token], min(self.k, len(self.icd_to_ndc[token]))))
        return list(set(outputs))
