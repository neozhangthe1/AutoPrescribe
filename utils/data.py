import pickle
import os


directory = "data/"
model_directory = "build/"

_PAD = b"_PAD"
_GO = b"_GO"
_EOS = b"_EOS"
_UNK = b"_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 4127
GO_ID = 4128
EOS_ID = 4129
UNK_ID = 4130


def get_path(path):
    return os.path.join(directory, path)


def get_model_path(path):
    return os.path.join(model_directory, path)


def clean_data():
    train_set = load("mimic_episodes_train.pkl")
    test_set = load("mimic_episodes_test.pkl")

    def clean(src_set):
        for pair in src_set:
            if "0" in pair[1]:
                pair[1].remove("0")
            if '' in pair[1]:
                pair[1].remove('')

    dump(train_set, "mimic_episodes_train.pkl")
    dump(test_set, "mimic_episodes_test.pkl")


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


def downgrade_pickle():
    for f in ["mimic_episodes_test.pkl", "mimic_episodes_train.pkl", "diag_vocab.pkl", "drug_vocab.pkl", "mimic_episodes.pkl"]:
        with open(f, "rb") as f_in:
            d = pickle.load(f_in)
            with open(f+'1', "wb") as f_out:
                pickle.dump(d, f_out, protocol=2)


def dump(obj, path):
    with open(get_path(path), "wb") as f_out:
        pickle.dump(obj, f_out)


def load(path):
    return pickle.load(open(get_path(path), "rb"))


if __name__ == "__main__":
    run_to_index()