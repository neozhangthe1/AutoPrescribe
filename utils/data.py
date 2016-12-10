import pickle
import os


directory = "data/"


def get_path(path):
    return os.path.join(directory, path)


def to_index(src_set, input_vocab, output_vocab):
    target_set = []
    for pair in src_set:
        inputs = []
        for token in pair[0]:
            if token in input_vocab:
                inputs.append(input_vocab[token])
        outputs = []
        for token in pair[1]:
            if token in output_vocab:
                outputs.append(output_vocab[token])
        target_set.append([inputs, outputs])
    return target_set


def run_to_index():
    train_set = pickle.load(open(get_path("mimic_episodes_train.pkl"), "rb"))
    test_set = pickle.load(open(get_path("mimic_episodes_test.pkl"), "rb"))
    input_vocab = pickle.load(open(get_path("diag_vocab.pkl"), "rb"))
    output_vocab = pickle.load(open(get_path("drug_vocab.pkl"), "rb"))
    train_index_set = to_index(train_set, input_vocab, output_vocab)
    test_index_set = to_index(test_set, input_vocab, output_vocab)
    with open(get_path("mimic_episodes_index_train.pkl"), "wb") as f_out:
        pickle.dump(train_index_set, f_out)
    with open(get_path("mimic_episodes_index_test.pkl"), "wb") as f_out:
        pickle.dump(test_index_set, f_out)


if __name__ == "__main__":
    run_to_index()