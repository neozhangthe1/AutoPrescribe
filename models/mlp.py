from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
import numpy as np


class MLP(object):
    def __init__(self):
        self.model = Sequential()
        self.train_x, self.train_y, self.test_x, self.test_y = [], [], [], []

    def build_model(self):
        self.model.add(Dense(output_dim=64, input_dim=100))
        self.model.add(Activation("relu"))
        self.model.add(Dense(output_dim=10))
        self.model.add(Activation("softmax"))

        self.model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01, momentum=0.9, nesterov=True))

    def load_data(self, train_set, test_set, source_size, target_size):
        self.train_x = np.zeros((len(train_set), source_size))
        self.train_y = np.zeros((len(train_set), target_size))
        self.test_x = np.zeros((len(test_set), source_size))
        self.test_y = np.zeros((len(test_set), target_size))
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

    def fit(self):
        self.model.fit(self.train_x, self.train_y, nb_epoch=5, batch_size=32)

    def eval(self):
        loss_and_metrics = self.model.evaluate(self.test_x, self.train_y, batch_size=32)
        return loss_and_metrics

    def predict(self, x):
        classes = self.model.predict_classes(x, batch_size=32)
        proba = self.model.predict_proba(x, batch_size=32)
        return classes, proba
