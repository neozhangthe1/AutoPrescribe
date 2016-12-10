import pickle


class Evaluator(object):

    def __init__(self):
        self.test_set = pickle.load(open("data/mimic_episodes_test.pkl", "rb"))
        self.vocab_input = pickle.load(open("data/diag_vocab.pkl", "rb"))
        self.vocab_output = pickle.load(open("data/drug_vocab.pkl", "rb"))

    def eval(self, model):
        for d in self.test_set:
            inputs = [self.vocab_input[token] for token in d[0]]
            outputs = set([self.vocab_output[token] for token in d[1]])
            prediction = set(model.predict(inputs))
            tp = len(outputs.intersection(prediction))
            fp = len(prediction - outputs)
            fn = len(outputs - prediction)
            prec = float(tp / (tp + fp))
            recall = float(tp / (tp + fn))
