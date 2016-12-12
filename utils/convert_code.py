from utils.icd9 import ICD9
from collections import defaultdict as dd
from utils.data import dump, load

tree = ICD9('data/icd9.json')


def get_leaves(code):
    if "-" in code:
        code = code.split(".")[0]
    node = tree.find(code)
    if node is None:
        print(code)
        if "-" in code:
            x = code.split("-")
            node = tree.find(x[0])
        else:
            node = tree.find("V" + code)
    if node is None:
        print(code, "!")
        return [code]
    return [n.code for n in node.leaves]


def build_ground_truth():
    rx_to_ndc = load("rx_to_ndc.npy")
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
                codes = get_leaves(icd)
                for code in codes:
                    icd_to_ndc[code].append(ndc)

    dump(icd_to_ndc, "icd_to_ndc.pkl")
