import csv
from collections import defaultdict as dd
import codecs
import pickle
from utils.data import dump, load
import random


def split_data(encounters):
    cleaned_encounters = []
    for d in encounters:
        if "" in d[1]:
            d[1].remove("")
        if len(d[1]) > 0:
            cleaned_encounters.append(d)
    random.shuffle(cleaned_encounters)
    encounters_train = cleaned_encounters[:2300000]
    encounters_dev = cleaned_encounters[2300000:]
    with open("sutter_encounter.train.pkl", "wb") as f_out:
        pickle.dump(encounters_train, f_out)
    with open("sutter_encounter.dev.pkl", "wb") as f_out:
        pickle.dump(encounters_dev, f_out)


def load_sutter():
    cnt = 0
    records = dd(lambda: dd(lambda: dd(list)))
    with codecs.open("SUTTER_ORDER_MED_DETAIL_V1.tab", "r", encoding='utf-8', errors='ignore') as f_in:
        next(f_in)
        for line in f_in:
            if cnt % 100000 == 0:
                print(cnt, len(records), line)
            cnt += 1
            x = line.strip().split("\t")
            records[x[0]][x[9]][x[1]].append(x)

    rec = dict(records)
    with open("sutter_prescription.pkl", "wb") as f_out:
        pickle.dump(rec, f_out)

    import json
    with open("sutter_prescription.json", "w") as f_ouut:
        json.dump(rec, f_out)


def get_pairs(data):
    meds = dd(list)
    for pid in data:
        for eid in data[pid]:
            for mid in data[pid][eid]:
                for med in data[pid][eid][mid]:
                    meds[eid].append((med[2], med[23]))


def get_prescription():
    meds = dd(set)
    cnt = 0
    with codecs.open("SUTTER_ORDER_MED_DETAIL_V1.tab", "r", encoding='utf-8', errors='ignore') as f_in:
        next(f_in)
        for line in f_in:
            if cnt % 100000 == 0:
                print(cnt, len(meds))
            cnt += 1
            x = line.strip().split("\t")
            meds[x[9]].add((x[2], x[12], x[13], x[-1]))


def get_encounter(eids):
    encounters = dd(list)
    cnt = 0
    with codecs.open("SUTTER_ENCOUNTER_DETAIL_V1.tab", "r", encoding='utf-8', errors='ignore') as f_in:
        next(f_in)
        for line in f_in:
            if cnt % 100000 == 0:
                print(cnt, len(encounters))
            cnt += 1
            x = line.strip().split("\t")
            encounters[x[1]].append(x[6])
    valid_encounters = {}
    for eid in eids:
        valid_encounters[eid] = encounters[eid]


def load_medication_details():
    medications = {}
    cnt = 0
    with codecs.open("SUTTER_MEDICATIONS_DETAIL_V1.tab", "r", encoding='utf-8', errors='ignore') as f_in:
        next(f_in)
        for line in f_in:
            if cnt % 100000 == 0:
                print(cnt, len(medications))
            cnt += 1
            x = line.split("\t")
            medications[x[0]] = x
    return medications


def join_encounters(valid_encounters, meds, medications):
    pairs = []
    for eid in valid_encounters:
        pairs.append((valid_encounters[eid], meds[eid]))
    diag_pres = []
    for p in pairs:
        pres = []
        for med in p[1]:
            med_detail = medications[med[-1]]
            pres.append(tuple(list(med)+[med_detail[2], med_detail[3], med_detail[6], med_detail[7]]))
        diag_pres.append((p[0], pres))
    dump(pairs, "diagnosis_prescription_pairs.pkl")


def search_encounter_by_drug(diag_pres, code):
    results = []
    for p in diag_pres:
        for med in p[1]:
            if code == med[-1]:
                results.append(p)
    return results


def join_prescription(meds, medications):
    prescriptions = []
    for eid in meds:
        pres = []
        for med in meds[eid]:
            if not med[-1] in medications:
                continue
            med_detail = medications[med[-1]]
            pres.append(tuple(list(med)+[med_detail[2], med_detail[3], med_detail[6], med_detail[7]]))
        prescriptions.append(pres)
    dump(prescriptions, "prescriptions.pkl")


def search_precription_by_drug(prescriptions, code):
    results = []
    for p in prescriptions:
        for med in p:
            if code == med[-1]:
                results.append(p)
                break
    for r in results:
        diag = set()
        drug = set()
        for item in r:
            diag.add(item[0])
            drug.add(item[1])
        print(" ".join(diag))
        print("\n".join(drug))
        print("\n")
    return results


def clean_encounter(diag_pres):
    encounters = []
    for p in diag_pres:
        pres = set()
        for med in p[1]:
            pres.add(med[-1])
        encounters.append((p[0], list(pres)))
    dump(encounters, "sutter_encounters.pkl")


def load_encounter():
    encounters1 = set()
    encounters2 = set()
    cnt = 0
    with codecs.open("SUTTER_ENCOUNTER_DETAIL_V1.tab", "r", encoding='utf-8', errors='ignore') as f_in:
        next(f_in)
        for line in f_in:
            if cnt % 100000 == 0:
                print(cnt, len(encounters1))
            cnt += 1
            x = line.strip().split("\t")
            encounters1.add(x[1])
    cnt = 0
    with codecs.open("SUTTER_ORDER_MED_DETAIL_V1.tab", "r", encoding='utf-8', errors='ignore') as f_in:
        next(f_in)
        for line in f_in:
            if cnt % 100000 == 0:
                print(cnt, len(encounters2))
            cnt += 1
            x = line.strip().split("\t")
            encounters2.add(x[9])

    dump(encounters1.intersection(encounters2), "valid_encounters")


def group_encounter_by_diag(diag_pres):
    encounter_by_diag = dd(list)
    for p in diag_pres:
        pres = set()
        for med in p[1]:
            pres.add(med[-1])
        encounter_by_diag[tuple(p[0])].append(pres)


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
    dump(new_encounters_clean, "sutter_encounters_%s.pkl" % level)
    dump(new_encounters_clean[:len(new_encounters_clean) * 0.8], "sutter_encounters_%s.train.pkl" % level)
    dump(new_encounters_clean[len(new_encounters_clean) * 0.8:], "sutter_encounters_%s.test.pkl" % level)
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
    dump(diag_vocab, "sutter_diag_vocab.pkl")
    dump(drug_vocab, "sutter_drug_vocab_%s.pkl" % level)


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


order_encounters("sutter_encounters_2.test")
order_encounters("sutter_encounters_4.train")
order_encounters("sutter_encounters_4.test")
order_encounters("sutter_encounters_6.train")
order_encounters("sutter_encounters_6.test")



def gen_parallel_text():
    f_out1 = open("sutter_diag.txt", "w")
    f_out2 = open("sutter_drug.txt", "w")
    encounters = load("sutter_encounters_3.pkl")
    for enc in encounters:
        f_out1.write(" ".join(enc[0]) + "\n")
        f_out2.write(" ".join(enc[1]) + "\n")
    f_out1.close()
    f_out2.close()


def sort_encounter():
    train = load("sutter_encounter.train.pkl")
    test = load("sutter_encounter.dev.pkl")
    sorted_train = []
    sorted_test = []
    for d in train:
        sorted_train.append((d[0], sorted(d[1])))
    for d in test:
        sorted_test.append((d[0], sorted(d[1])))
    dump(sorted_train, "sutter_encounter_sorted.train.pkl")
    dump(sorted_test, "sutter_encounter_sorted.dev.pkl")


def extract_mapping():
    medications = load_medication_details()
    cnt = 0
    records = {}
    with codecs.open("SUTTER_ORDER_MED_DETAIL_V1.tab", "r", encoding='utf-8', errors='ignore') as f_in:
        next(f_in)
        for line in f_in:
            if cnt % 100000 == 0:
                print(cnt, len(records), line)
            cnt += 1
            x = line.strip().split("\t")
            if x[-1] in medications and medications[x[-1]][7] != "":
                records[x[1]] = (x[2], medications[x[-1]][7])

    diag_cnt = dd(int)
    diag_drug_pair_cnt = dd(int)
    drug_cnt = dd(int)
    for diag, drug in records.values():
        diag_drug_pair_cnt[(diag, drug[:6])] += 1
        diag_cnt[diag] += 1
        drug_cnt[drug[:6]] += 1


    sorted_diag_drug_pair = sorted(diag_drug_pair_cnt.items(),key=lambda x: x[1], reverse=True)

    diag_to_drug = dd(list)
    drug_to_diag = dd(list)
    for diag, drug in diag_drug_pair_cnt:
        diag_to_drug[diag].append(drug)
        drug_to_diag[drug].append(diag)
    dump((dict(diag_to_drug), dict(drug_to_diag)), "diag_drug_mapping.pkl")