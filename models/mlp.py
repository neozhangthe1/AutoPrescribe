from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
from keras.models import load_model
import numpy as np
from utils.data import get_model_path, load
from sklearn import metrics


class MLP(object):
    def __init__(self, data="sutter"):
        self.model = Sequential()
        self.train_x, self.train_y, self.test_x, self.test_y = [], [], [], []
        self.input_dim = 0
        self.output_dim = 0
        self.data = data

    def build_model(self):
        self.model.add(Dense(output_dim=2000, input_dim=self.input_dim))
        self.model.add(Activation("relu"))
        self.model.add(Dense(output_dim=1000, input_dim=self.input_dim))
        self.model.add(Activation("relu"))
        self.model.add(Dense(output_dim=1000, input_dim=self.input_dim))
        self.model.add(Activation("relu"))
        # self.model.add(Dense(output_dim=2000, input_dim=self.input_dim))
        # self.model.add(Activation("relu"))
        self.model.add(Dense(output_dim=self.output_dim))
        self.model.add(Activation("softmax"))

        self.model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01, momentum=0.9, nesterov=True), metrics=['accuracy', 'precision', 'recall'])

    def load_data(self, train_set, test_set, source_size, target_size):
        self.train_x = np.zeros((len(train_set), source_size))
        self.train_y = np.zeros((len(train_set), target_size))
        self.test_x = np.zeros((len(test_set), source_size))
        self.test_y = np.zeros((len(test_set), target_size))
        self.input_dim = source_size
        self.output_dim = target_size
        for i, pair in enumerate(train_set):
            for j in pair[0]:
                self.train_x[i, j] = 1
            for j in pair[1]:
                self.train_y[i, j] = 1
        for i, pair in enumerate(test_set):
            for j in pair[0]:
                self.test_x[i, j] = 1
            for j in pair[1]:
                self.test_y[i, j] = 1

    def fit(self, epoch):
        self.model.fit(self.train_x, self.train_y, validation_data=(self.test_x, self.test_y), nb_epoch=epoch, batch_size=32, verbose=True)
        self.model.save(self.data + "_mlp_model.h5")

    def eval(self):
        loss_and_metrics = self.model.evaluate(self.test_x, self.test_y, batch_size=32)
        return loss_and_metrics

    def predict(self, x):
        classes = self.model.predict_classes(x, batch_size=32)
        proba = self.model.predict_proba(x, batch_size=32)
        return classes, proba


def train():
    input_vocab = load("sutter_diag_vocab.pkl")
    output_vocab = load("sutter_drug_vocab_3.pkl")
    train_encounters = load("sutter_encounter.train.pkl")
    test_encounters  = load("sutter_encounter.dev.pkl")
    test_set = []
    train_set = []
    for enc in train_encounters:
        train_set.append(([input_vocab[code] for code in enc[0]], [output_vocab[code] for code in enc[1]]))
    for enc in test_encounters:
        test_set.append(([input_vocab[code] for code in enc[0]], [output_vocab[code] for code in enc[1]]))
    mlp = MLP()
    mlp.load_data(train_set, test_set[:1000], len(input_vocab), len(output_vocab))
    mlp.build_model()
    mlp.fit(1)
    mlp.predict(test_set)

def train_mimic():
    input_vocab = load("mimic_diag_vocab.pkl")
    output_vocab = load("mimic_drug_vocab.pkl")
    train_encounters = load("mimic_encounter_gpi.train.pkl")
    test_encounters  = load("mimic_encounter_gpi.dev.pkl")
    test_set = []
    train_set = []
    for enc in train_encounters:
        train_set.append(([input_vocab[code] for code in enc[0]], [output_vocab[code] for code in enc[1]]))
    for enc in test_encounters:
        test_set.append(([input_vocab[code] for code in enc[0]], [output_vocab[code] for code in enc[1]]))
    mlp = MLP("mimic")
    mlp.load_data(train_set, test_set, len(input_vocab), len(output_vocab))
    mlp.build_model()
    mlp.fit(5)
    mlp.predict(test_set)

def test():
    input_vocab = load("sutter_diag_vocab.pkl")
    output_vocab = load("sutter_drug_vocab_3.pkl")
    test_encounters  = load("sutter_encounter.dev.pkl")
    test_set = []
    for enc in test_encounters:
        test_set.append(([input_vocab[code] for code in enc[0]], [output_vocab[code] for code in enc[1]]))
    mlp = load_model("sutter_mlp_model.h5")

    input_dim = len(input_vocab)
    output_dim = len(output_vocab)
    test_x = np.zeros((len(test_set), input_dim))
    test_y = np.zeros((len(test_set), output_dim))

    for i, pair in enumerate(test_set):
        for j in pair[0]:
            test_x[i, j] = 1
        for j in pair[1]:
            test_y[i, j] = 1

    index_to_source = {}
    index_to_target = {}
    for token in input_vocab:
        index_to_source[input_vocab[token]] = token
    for token in output_vocab:
        index_to_target[output_vocab[token]] = token

    for i in range(1, 10):
        threshold = float(i) / 500.0
        labels, results = mlp.predict(test_x)

        results[results >= threshold] = 1
        results[results < threshold] = 0

        jaccard = metrics.jaccard_similarity_score(test_y, results)
        print(threshold, jaccard)

    labels, results = mlp.predict(test_x)
    results[results >= 0.012] = 1
    results[results < 0.012] = 0
    cnts, indices = results.nonzero()
    jaccard = metrics.jaccard_similarity_score(test_y, results)
    zero_one = metrics.jaccard_similarity_score(test_y, results)


    outputs = [[] for i in range(len(test_set))]
    for i, cnt in enumerate(cnts):
        outputs[cnt].append(index_to_target[indices[i]])

    merge = []
    for i, item in enumerate(outputs):
        print(test_encounters[i][0])
        print(test_encounters[i][1])
        print(outputs[i])
        print("")

        merge.append(list(test_encounters[i]) + [outputs[i]])

    from utils.data import dump
    dump(merge, "mimic_result_mlp_0.012.pkl")

    truth_list = []
    prediction_list = []
    for enc in merge:
        truth_list.append(enc[1])
        prediction_list.append(enc[2])



