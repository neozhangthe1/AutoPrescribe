from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
from keras.models import load_model
import numpy as np
from utils.data import get_model_path, load


class MLP(object):
    def __init__(self):
        self.model = Sequential()
        self.train_x, self.train_y, self.test_x, self.test_y = [], [], [], []
        self.input_dim = 0
        self.output_dim = 0

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
        self.model.fit(self.train_x, self.train_y, validation_data=(self.test_x, self.test_y), nb_epoch=50, batch_size=32, verbose=True)

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
    encounters = load("sutter_encounters_3.pkl")
    test_set = []
    train_set = []
    for enc in encounters[:1000000]:
        train_set.append(([input_vocab[code] for code in enc[0]], [output_vocab[code] for code in enc[1]]))
    for enc in encounters[1000000:]:
        test_set.append(([input_vocab[code] for code in enc[0]], [output_vocab[code] for code in enc[1]]))
    mlp = MLP()
    mlp.load_data(train_set, test_set, len(input_vocab), len(output_vocab))
    mlp.build_model()
    mlp.fit(1)

def test():
    input_vocab = load("sutter_diag_vocab.pkl")
    output_vocab = load("sutter_drug_vocab_3.pkl")
    encounters = load("sutter_encounters_3.pkl")
    test_set = []
    train_set = []
    for enc in encounters[:1000000]:
        train_set.append(([input_vocab[code] for code in enc[0]], [output_vocab[code] for code in enc[1]]))
    for enc in encounters[1000000:]:
        test_set.append(([input_vocab[code] for code in enc[0]], [output_vocab[code] for code in enc[1]]))
    mlp = load_model("mlp_sutter.model")

    input_dim = len(input_vocab)
    output_dim = len(output_vocab)
    test_x = np.zeros((len(test_set), input_dim))
    test_y = np.zeros((len(test_set), output_dim))

    for i, pair in enumerate(test_set):
        for j in pair[0]:
            test_x[i, j] = 1
        for j in pair[1]:
            test_y[i, j] = 1
    for item in test_set:
    mlp.load_data(train_set, test_set, len(input_vocab), len(output_vocab))
    mlp.build_model()
    mlp.fit(1)
