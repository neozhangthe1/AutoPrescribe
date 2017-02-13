from utils.data import load, dump
from collections import defaultdict as dd
import numpy as np

# from models import MostFreqMatch, Embedding, GoldenRule


class Evaluator(object):

    def __init__(self, ds="mimic"):
        if ds == "mimic":
            self.test_set = load("mimic_episodes_index_test.pkl")
            self.golden_rule = load("icd_to_ndc_index.pkl")
            diag_vocab = load("diag_vocab.pkl")
            drug_vocab = load("drug_vocab.pkl")
        elif ds == "sutter":
            self.test_set = load("sutter_encounters_3.pkl")
            self.golden_rule = load("icd_to_ndc_index.pkl")
            diag_vocab = load("sutter_diag_vocab.pkl")
            drug_vocab = load("sutter_drug_vocab_3.pkl")
        self.index_to_diag = {}
        self.index_to_drug = {}
        for diag in diag_vocab:
            self.index_to_diag[diag_vocab[diag]] = diag
        for drug in drug_vocab:
            self.index_to_drug[drug_vocab[drug]] = drug

    def eval(self, model):
        precisions, recalls, jaccards = [], [], []
        cnt = 0
        for pair in self.test_set:
            outputs = set(pair[1])
            prediction = set(model.predict(pair[0]))
            precision, recall, jaccard = self.get_result(outputs, prediction)
            precisions.append(precision)
            recalls.append(recall)
            jaccards.append(jaccard)
            if cnt % 1000 == 0:
                print(cnt, precision, recall, jaccard)
                print(np.mean(precisions), np.mean(recalls), np.mean(jaccards))
            cnt += 1
        print(np.mean(precisions), np.mean(recalls), np.mean(jaccards))

    def eval_golden(self, model):
        precisions, recalls = [], []
        cnt = 0
        for pair in self.test_set:
            prediction = model.predict(pair[0])
            precision, recall = self.get_golden_eval(pair[0], prediction)
            precisions.append(precision)
            recalls.append(recall)
            if cnt % 1000 == 0:
                print(cnt, precision, recall)
                print(np.mean(precisions), np.mean(recalls))
            cnt += 1
        print(np.mean(precisions), np.mean(recalls))

    @staticmethod
    def get_result(truth, prediction):
        truth = set(truth)
        prediction = set(prediction)
        tp = len(truth.intersection(prediction))
        fp = len(prediction - truth)
        fn = len(truth - prediction)
        precision = 0 if (tp + fp) == 0 else float(tp / (tp + fp))
        recall = 0 if (tp + fn) == 0 else float(tp / (tp + fn))
        jaccard = 0 if len(truth.intersection(prediction)) == 0 else float(len(truth.intersection(prediction))) / len(truth.union(prediction))

        return precision, recall, jaccard
    
def get_golden_eval(inputs, prediction, diag_to_drug, drug_to_diag):
    prediction = set(prediction)
    all_valid_output = []
    for c1 in inputs:
        if c1 in diag_to_drug:
            all_valid_output.extend(diag_to_drug[c1])
    all_valid_output = set(all_valid_output)
    tp = prediction.intersection(all_valid_output)

    precision = 0 if len(prediction) == 0 else float(len(tp)) / len(prediction)

    all_valid_input = []
    for c in prediction:
        if c in drug_to_diag:
            all_valid_input.extend(drug_to_diag[c])
    tp = set(all_valid_input).intersection(set(inputs))
    recall = 0 if len(inputs) == 0 else float(len(tp)) / len(inputs)

    return precision, recall

def get_average_golden_eval(input_list, prediction_list):
    diag_to_drug, drug_to_diag = load("icd_gpi_map.pkl")
    ave_precision, ave_recall = 0.0, 0.0
    for i, kk in enumerate(input_list):
        item = [k.replace(".", "") for k in kk]
        precision, recall = get_golden_eval(item, prediction_list[i], diag_to_drug, drug_to_diag)
        ave_precision += precision
        ave_recall += recall
        if i % 1000 == 0:
            print(ave_precision / (i+1), ave_recall / (i+1), precision, recall)

def evaluate(name):
    results = load(name)
    input_list, truth_list, prediction_list = [], [], []
    for i, result in enumerate(results):
        input_list.append(result[0])
        truth_list.append(result[1])
        prediction_list.append(result[2])


def get_jaccard_k(truth, prediction, k=1):
    import itertools

    def get_set_product(set1, kk):
        results = set()
        if kk == 2:
            for item in itertools.product(set1, set1):
                if len(set(item)) == len(item):
                    results.add(tuple(sorted(item)))
        elif kk == 3:
            for item in itertools.product(set1, set1, set1):
                if len(set(item)) == len(item):
                    results.add(tuple(sorted(item)))
        return results

    s1 = set(truth)
    s2 = set(prediction)
    if k > 1:
        for item in get_set_product(truth, 2):
            s1.add(item)
        for item in get_set_product(prediction, 2):
            s2.add(item)
    if k > 2:
        for item in get_set_product(truth, 3):
            s1.add(item)
        for item in get_set_product(prediction, 3):
            s2.add(item)
    interaction = len(s1.intersection(s2))
    union = len(s1.union(s2))
    if union == 0:
        return 0
    return float(interaction) / union

def get_average_jaccard(truth_list, prediction_list):
    jaccard = 0.0
    cnt = 0
    for i, item in enumerate(prediction_list):
        jaccard += get_jaccard_k(truth_list[i], item)
        cnt += 1
    print(jaccard / cnt)


def get_accuracy(truth, prediction):
    if set(prediction) == set(truth):
        return 1.
    else:
        return 0.


def get_average_accuracy(truth_list, prediction_list):
    acc = 0.0
    cnt = 0
    for i, item in enumerate(prediction_list):
        acc += get_accuracy(truth_list[i], item)
        cnt += 1
    print(acc / cnt)

def get_macro_f1(truth_list, prediction_list):
    tp = dd(float)
    true = dd(float)
    predict = dd(float)
    for i, item in enumerate(prediction_list):
        for token in item:
            predict[token] += 1
            if token in truth_list[i]:
                tp[token] += 1
        for token in truth_list[i]:
            true[token] += 1
    precision = {}
    recall = {}
    f1 = {}
    for c in true:
        if c in predict:
            precision[c] = tp[c] / predict[c]
        else:
            precision[c] = 0
        recall[c] = tp[c] / true[c]
        f1[c] = 0 if (precision[c] + recall[c]) == 0 else 2 * (precision[c] * recall[c]) / (precision[c] + recall[c])
    print(np.average(list(f1.values())))


def eval_interaction(prediction_list):
    interaction_drugs = load("interaction_drugs.pkl")
    neg_count = 0
    total_count = 0
    for pd in prediction_list:
        prediction = list(pd)
        flag = False
        total_count += len(prediction) * (len(prediction) - 1)
        for i1 in range(len(prediction)):
            for i2 in range(i1+1, len(prediction)):
                if (prediction[i1], prediction[i2]) in interaction_drugs:
                    print((prediction[i1], prediction[i2]), neg_count)
                    neg_count += 1
                    # flag = True
                    # break
            # if flag:
            #     break
        # if flag:
        #     neg_count += 1
        print(neg_count, len(prediction_list), total_count)
        print(0 if total_count == 0 else float(neg_count) / total_count)


def merge():
    gt = dd(set)
    result_names = ["sutter_sorted_result_seq2seq.pkl", "sutter_result_seq2seq_1.30.pkl", "sutter_result_mlp_0.15.pkl", "sutter_result_freq.pkl"]
    # result_names = ["mimic_ontology_seq2seq.pkl", "mimic_result_freq.pkl", "mimic_result_mlp_0.012.pkl", "mimic_result_seq2seq_3_80.pkl", "mimic_sorted_result_seq2seq_3_80.pkl", "mimic_unsort_result_seq2seq.pkl"]
    results = [load(r) for r in result_names]

    merged_results = [dd(list) for _ in range(len(results))]
    for item in results[0]:
        item_0 = [x.replace(".", "") for x in item[0]]
        gt[tuple(sorted(item_0))].add(tuple(sorted(item[1])))
    sorted_gt = sorted(gt.items(), key=lambda x: len(x[1]), reverse=True)
    for i, result in enumerate(results):
        for item in result:
            item_0 = [x.replace(".", "") for x in item[0]]
            merged_results[i][tuple(sorted(item_0))].append(item[2])
    final_results = {}
    for r in sorted_gt:
        final_results[r[0]] = [list(set(r[1]))]
        for result in merged_results:
            final_results[r[0]].append(result[r[0]][:1])

    inputs = "E66.9,311,477.8,493.9".split(",")
    def get_results(inputs):
        to_retrieve = []
        inputs = tuple(sorted([x.replace(".", "") for x in inputs]))
        for code in final_results:
            if len(set(code).intersection(set(inputs))) > 0:
                to_retrieve.append(code)
        return to_retrieve
        # print(final_results[inputs])

    to_retrieve = get_results(inputs)

    drug_name = load("drug_name.pkl")
    diag_name = load("diag_name.pkl")

    with open("gt_sutter.txt", "w") as f_out:
        for item in results[0]:
            f_out.write("\t".join([diag_name[x] + " " + x for x in item[0]]) + "\n")
            f_out.write("\t".join(sorted(drug_name[x] + " " + x for x in item[1])) + "\n")
            f_out.write("\n")

    encounters = load("sutter_encounter_clean.train.pkl")

    with open("gt_sutter.txt", "w") as f_out:
        for item in encounters[:10000]:
            f_out.write(",".join([x + " " + diag_name[x].replace(",", " ") for x in item[0]]) + "\n")
            f_out.write(",".join(sorted(x + " " + drug_name[x].replace(",", " ") for x in item[1])) + "\n")
            f_out.write("\n")

    with open("gt_sutter.txt", "w") as f_out:
        for item in encounters[:10000]:
            for i, x in enumerate(item[0]):
                if i == 0:
                    f_out.write("Diagnosis:,")
                else:
                    f_out.write(",")
                f_out.write(x + "," + diag_name[x].replace(",", " ") + "\n")
            for i, x in enumerate(item[1]):
                if i == 0:
                    f_out.write("Drug:,")
                else:
                    f_out.write(",")
                f_out.write(x + "," + drug_name[x].replace(",", " ") + "\n")
            f_out.write("\n")

    with open("sutter_result_1.csv", "w") as f_out:
        for k in to_retrieve:
            r = final_results[k]
            for i, x in enumerate(k):
                if i == 0:
                    f_out.write("Diagnosis_%s:," % i)
                else:
                    f_out.write(",")
                f_out.write(x + "," + diag_name[x].replace(",", " ") + "\n")
            # prescriptions = set()
            for j, item in enumerate(r):
                item_0 = []
                if len(item) > 0:
                    item_0 = item[0]
                for i, x in enumerate(item_0):
                    if i == 0:
                        f_out.write("Method_%s:," % j)
                    else:
                        f_out.write(",")
                    f_out.write(x + "," + drug_name[x].replace(",", " ") + "\n")
            f_out.write("\n")



    with open("sutter_results.csv", "w") as f_out:
        for k, r in enumerate(final_results):
            for i, x in enumerate(r):
                if i == 0:
                    f_out.write("Diagnosis_%s:," % k)
                else:
                    f_out.write(",")
                f_out.write(x + "," + diag_name[x].replace(",", " ") + "\n")
            prescriptions = set()
            for j, item in enumerate(final_results[r]):
                if len(item) > 0:
                    prescriptions.add(tuple(sorted(item[0])))
            for j, item in enumerate(prescriptions):
                for i, x in enumerate(item):
                    if i == 0:
                        f_out.write("Drug_%s:," % j)
                    else:
                        f_out.write(",")
                    f_out.write(x + "," + drug_name[x].replace(",", " ") + "\n")
            f_out.write("\n")

    with open("mimic_results_all.csv", "w") as f_out:
        for k, r in enumerate(final_results):
            for i, x in enumerate(r):
                if i == 0:
                    f_out.write("Diagnosis_%s:," % k)
                else:
                    f_out.write(",")
                if x in diag_name:
                    f_out.write(x + "," + diag_name[x].replace(",", " ") + "\n")
            prescriptions = []
            for j, item in enumerate(final_results[r]):
                if len(item) > 0:
                    prescriptions.append(tuple(sorted(item[0])))
            for j, item in enumerate(prescriptions):
                for i, x in enumerate(item):
                    if i == 0:
                        f_out.write("Drug_%s:," % j)
                    else:
                        f_out.write(",")
                    f_out.write(x + "," + drug_name[x].replace(",", " ") + "\n")
            f_out.write("\n")


