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