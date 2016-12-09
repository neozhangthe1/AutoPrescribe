from gensim.models import Word2Vec
import pickle


model = Word2Vec()


def run():
    train_data = pickle.load(open("mimic_episodes_test.pkl", "rb"))
    test_data = pickle.load(open("mimic_episodes_train.pkl", "rb"))

    sentences = []
    for d in train_data:
        for t1 in d[0]:
            for t2 in d[1]:
                sentences.append([t1, t2])
    model = Word2Vec(sentences, size=100, window=5, min_count=5, workers=4)
    model.save("mimic.emb")

