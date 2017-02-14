from utils.data import load, dump
from collections import defaultdict as dd
import random

def clean_mimic(ndc_to_gpi):
    mimic_data = load("mimic_episodes.pkl")
    clean_mimic_data = []
    for d in mimic_data:
        drugs = []
        for drug in d[1]:
            if len(drug) != 11:
                continue
            drugs.append(drug)
        clean_mimic_data.append((d[0], drugs))
    random.shuffle(clean_mimic_data)
    mimic_train = clean_mimic_data[:49000]
    mimic_dev = clean_mimic_data[49000:]
    dump(mimic_train, "mimic_encounter.train.pkl")
    dump(mimic_dev, "mimic_encounter.dev.pkl")

    mimic_data_gpi = []
    for d in clean_mimic_data:
        drugs = []
        for drug in d[1]:
            if drug in ndc_to_gpi:
                drugs.append(ndc_to_gpi[drug])
            else:
                print(drug)
        mimic_data_gpi.append((d[0], list(set(drugs))))
    dump(mimic_data_gpi, "mimic_encounter_gpi.pkl")
    dump(mimic_data_gpi[:49000], "mimic_encounter_gpi.train.pkl")
    dump(mimic_data_gpi[49000:], "mimic_encounter_gpi.dev.pkl")

    mimic_diag_vocab = {}
    mimic_drug_vocab = {}
    for d in mimic_data_gpi:
        for diag in d[0]:
            if diag not in mimic_diag_vocab:
                mimic_diag_vocab[diag] = len(mimic_diag_vocab)
        for drug in d[1]:
            if drug not in mimic_drug_vocab:
                mimic_drug_vocab[drug] = len(mimic_drug_vocab)
    dump(mimic_diag_vocab, "mimic_diag_vocab.pkl")
    dump(mimic_drug_vocab, "mimic_drug_vocab.pkl")


def load_ndc_gpi_mapping():
    f_in = open("data/ndw_v_product.txt")
    title = next(f_in).strip().split("|")
    data = []
    for line in f_in:
        data.append(line.strip().split("|"))

    ndc_to_gpi_6 = {}
    for d in data:
        ndc_to_gpi_6[d[1]] = d[58]

    dump(ndc_to_gpi_6, "ndc_to_gpi_6.pkl")

def load_mapping():
    diag_drug_mapping = load("diag_drug_mapping.pkl")
    diag_to_drug = {}
    drug_to_diag = {}
    for diag in diag_drug_mapping[0]:
        diag_to_drug[diag.replace(".", "")] = diag_drug_mapping[0][diag]
    for drug in diag_drug_mapping[1]:
        drug_to_diag[drug] = []
        for diag in diag_drug_mapping[1][drug]:
            drug_to_diag[drug].append(diag.replace(".", ""))
    dump((diag_to_drug, drug_to_diag), "mimic_diag_drug_mapping.pkl")

def sort_encounter():
    train = load("mimic_encounter_gpi.train.pkl")
    test = load("mimic_encounter_gpi.dev.pkl")
    sorted_train = []
    sorted_test = []
    for d in train:
        sorted_train.append((d[0], sorted(d[1])))
    for d in test:
        sorted_test.append((d[0], sorted(d[1])))
    dump(sorted_train, "mimic_encounter_gpi_sorted.train.pkl")
    dump(sorted_test, "mimic_encounter_gpi_sorted.dev.pkl")



def get_encounter_level(encounters, level, sorted_diag_rank):
    new_encounters = []
    for enc in encounters:
        input = []
        output = []
        for code in enc[0]:
            if len(code) > 0:
                input.append(code.replace(".", ""))
        for code in enc[1]:
            if len(code) > 0:
                output.append(code[:level])
        new_encounters.append((input, output))
    new_encounters_clean = clean_encounters(new_encounters, sorted_diag_rank)
    print(len(new_encounters_clean), len(new_encounters))
    dump(new_encounters_clean, "mimic_encounters_%s.pkl" % level)
    dump(new_encounters_clean[:int(len(new_encounters_clean) * 0.8)], "mimic_encounters_%s.train.pkl" % level)
    dump(new_encounters_clean[int(len(new_encounters_clean) * 0.8):], "mimic_encounters_%s.test.pkl" % level)
    gen_vocab(new_encounters_clean, level)


def get_freq(encounters):
    diag_count = dd(int)
    for enc in encounters:
        for code in enc[0]:
            diag_count[code] += 1
    sorted_diag_count = sorted(diag_count.items(), key=lambda x: x[1], reverse=True)
    sorted_diag_rank = {}
    for i, (code, freq) in enumerate(sorted_diag_count):
        sorted_diag_rank[code] = i
    return sorted_diag_rank


def clean_encounters(encounters, sorted_diag_rank):
    cnt = 0
    new_encounters = []
    for i, enc in enumerate(encounters):
        flag = True
        for code in enc[0]:
            if sorted_diag_rank[code] > 2000:
                flag = False
                cnt += 1
                break
        if flag:
            new_encounters.append(enc)
    return new_encounters


def gen_vocab(encounters, level):
    diag_vocab = {}
    drug_vocab = {}
    cnt1 = 0
    cnt2 = 0
    for p in encounters:
        for diag in p[0]:
            if not diag in diag_vocab:
                diag_vocab[diag] = cnt1
                cnt1 += 1
        for drug in p[1]:
            if not drug[:level] in drug_vocab:
                drug_vocab[drug[:level]] = cnt2
                cnt2 += 1
    dump(diag_vocab, "mimic_diag_vocab.pkl")
    dump(drug_vocab, "mimic_drug_vocab_%s.pkl" % level)


def order_encounters(name):
    import random
    print(name)
    encounters = load(name + '.pkl')
    orders = ["voc", "random", "freq", 'rare']
    ordered = [[] for _ in range(len(orders))]
    counters = dd(int)
    for enc in encounters:
        for code in enc[1]:
            counters[code] += 1
    vocab = sorted(list(counters.keys()))
    code_to_vocab_index = {}
    for i in range(len(vocab)):
        code_to_vocab_index[vocab[i]] = i
    for enc in encounters:
        if len(enc[1]) == 0:
            continue
        for i, order in enumerate(orders):
            enc_1 = list(set(enc[1]))
            if order == "voc":
                enc_1 = sorted(enc_1, key=lambda x: code_to_vocab_index[x])
                ordered[i].append((enc[0], enc_1))
            elif order == "random":
                random.shuffle(enc_1)
                ordered[i].append((enc[0], enc_1))
            elif order == "freq":
                enc_1 = sorted(enc_1, key=lambda x: counters[x], reverse=True)
                ordered[i].append((enc[0], enc_1))
            elif order == "rare":
                enc_1 = sorted(enc_1, key=lambda x: counters[x])
                ordered[i].append((enc[0], enc_1))
    for i, order in enumerate(orders):
        dump(ordered[i], name+"_"+order+".pkl")

order_encounters("mimic_encounters_2.train")
order_encounters("mimic_encounters_2.test")
order_encounters("mimic_encounters_4.train")
order_encounters("mimic_encounters_4.test")
order_encounters("mimic_encounters_6.train")
order_encounters("mimic_encounters_6.test")