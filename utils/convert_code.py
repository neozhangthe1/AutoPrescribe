from utils.icd9 import ICD9L, ICD9
from collections import defaultdict as dd
from utils.data import dump, load, get_path
import csv
import codecs


tree = ICD9("data/icd9.json")
icd9 = ICD9L('data/icd9.txt')
cache = {}


def get_leaves(code):
    if "-" in code:
        code = code.split(".")[0]

    if code in cache:
        return cache[code]

    if "|" in code:
        res = [get_leaves(c) for c in code.split("|")]
        cache[code] = res[0] + res[1]
        return cache[code]

    node = tree.find(code)
    if node is None:
        print(code)
        if "-" in code:
            x = code.split("-")
            node = tree.find(x[0])
        else:
            node = tree.find("0" + code)
    if node is None:
        print(code, "!")
        cache[code] = [code]
        return cache[code]
    cache[code] = [n.code for n in node.leaves]
    if "-" not in code:
        cache[code] += [code]
    return cache[code]


def normalize_icd(code):
    if "." in code:
        x = code.split(".")
        if len(x[0]) < 3:
            code = "0" * (3-len(x[0])) + code
        code = code.replace(".", "")
    return code


def load_rx_to_ndc():
    drugs = {}
    in_to_drug = dd(list)
    in_to_pin = dd(list)
    with codecs.open(get_path("rxnorm.csv"), "r", "utf-8") as f_in:
        reader = csv.reader(f_in)
        cnt = 0
        for row in reader:
            if cnt != 0:
                drug = {
                    "rx": row[0],
                    "tty": row[1],
                    "ndc": [],
                    "name": row[3],
                    "va_classes": row[4],
                    "treating": row[5].split(";"),
                    "ingredients": row[6].split(";")
                }
                if row[2] != '':
                    for code in row[2].strip("[").strip("]").split(","):
                        if code is not None and code != 'None':
                            drug["ndc"].append(code.strip().strip("'"))
                drugs[row[0]] = drug

                for ing in drug["ingredients"]:
                    in_to_drug[ing].append(drug)
                    if drug["tty"] == "PIN":
                        in_to_pin[ing].append(drug["rx"])
            cnt += 1

    rx_to_ndc = dd(list)
    for rx in drugs:
        for ndc in drugs[rx]["ndc"]:
            rx_to_ndc[rx].append(ndc)

    for ing in in_to_drug:
        for drug in in_to_drug[ing]:
            for ndc in drug["ndc"]:
                rx_to_ndc[ing].append(ndc)

    for ing in in_to_pin:
        for pin in in_to_pin[ing]:
            for ndc in rx_to_ndc[ing]:
                rx_to_ndc[pin].append(ndc)

    ndc_to_rx = dd(list)
    for rx in rx_to_ndc:
        for ndc in rx_to_ndc[rx]:
            ndc_to_rx[ndc].append(rx)

    dump(rx_to_ndc, "rx_to_ndc.pkl")
    dump(ndc_to_rx, "ndc_to_rx.pkl")


def build_ground_truth():
    rx_to_ndc = load("rx_to_ndc.pkl")
    rx_to_icd = dd(list)
    icd_to_ndc = dd(list)

    with open("./data/MEDI_11242015.csv") as f_in:
        cnt = -1
        for line in f_in:
            cnt += 1
            if cnt == 0:
                continue
            x = line.strip().split(",")
            rx_to_icd[x[0]].append(x[5])

    for rx in set(rx_to_icd.keys()).intersection(set(rx_to_ndc.keys())):
        for ndc in rx_to_ndc[rx]:
            for icd in rx_to_icd[rx]:
                codes = icd9.get_children(icd)
                for code in codes:
                    icd_to_ndc[code].append(ndc)
    for icd in icd_to_ndc:
        icd_to_ndc[icd] = list(set(icd_to_ndc[icd]))
    dump(dict(icd_to_ndc), "icd_to_ndc.pkl")

    ndc_to_icd = dd(list)
    for icd in icd_to_ndc:
        for ndc in icd_to_ndc[icd]:
            ndc_to_icd[ndc].append(icd)
    for ndc in ndc_to_icd:
        ndc_to_icd[ndc] = list(set(ndc_to_icd[ndc]))
    dump(dict(ndc_to_icd), "ndc_to_icd.pkl")


def convert_ground_truth():
    gt = load("icd_to_ndc.pkl")
    diag_vocab = load("diag_vocab.pkl")
    drug_vocab = load("drug_vocab.pkl")
    gt_index = {}
    for c in gt:
        code = normalize_icd(c)
        if code in diag_vocab:
            diag = diag_vocab[code]
            gt_index[diag] = []
            for code1 in gt[c]:
                if code1 in drug_vocab:
                    drug = drug_vocab[code1]
                    gt_index[diag].append(drug)
    for icd in gt_index:
        gt_index[icd] = list(set(gt_index[icd]))
    dump(gt_index, "icd_to_ndc_index.pkl")

    ndc_to_icd = dd(list)
    for icd in gt_index:
        for ndc in gt_index[icd]:
            ndc_to_icd[ndc].append(icd)
    for ndc in ndc_to_icd:
        ndc_to_icd[ndc] = list(set(ndc_to_icd[ndc]))
    dump(dict(ndc_to_icd), "ndc_to_icd_index.pkl")


