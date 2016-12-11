from gensim.models import Word2Vec
from utils.data import load, dump, get_path, get_model_path
import random


model = Word2Vec()


class Embedding(object):

    def __init__(self):
        self.model = None
        self.k = 3
        self.sample_size = 10
        self.samples = []

    def fit(self, train_set):
        samples = []
        for d in train_set:
            inputs = ["i_%s" % tk for tk in d[0]]
            outputs = ["o_%s" % tk for tk in d[1]]
            tokens = inputs + outputs
            if len(tokens) >= self.sample_size:
                for i in range(len(tokens)):
                    samples.append(random.sample(tokens, self.sample_size))
            else:
                for i in range(len(tokens)):
                    samples.append(random.sample(tokens, len(tokens)))
            samples.append(inputs)
            samples.append(outputs)
        self.samples = samples
        self.model = Word2Vec(samples, size=100, window=5, min_count=5, workers=4)
        self.model.save(get_model_path("mimic.emb"))

    def load(self):
        self.model = Word2Vec.load(get_model_path("mimic.emb"))

    def predict(self, inputs):
        outputs = []
        for token in inputs:
            tk = "i_%s" % token
            if tk in self.model:
                similar = self.model.most_similar(tk, topn=200)
                cnt = 0
                for tk1 in similar:
                    if tk1[0][0] == "o":
                        outputs.append(int(tk1[0].split("_")[1]))
                        cnt += 1
                        if cnt >= self.k:
                            break
        return list(set(outputs))
