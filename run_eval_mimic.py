from models.processor import Processor
from models.coverage import CoverageModel
from exp.coverage import config_mimic as config
from utils.data import dump

config = config.get_config()

print(config.saved_model_file.split('/')[-1])
p = Processor(config)
model = CoverageModel(p, config)

# model.do_train()

model.load_params('build/mimic_seq2seq_seed13_100d_lr0.001_h256.model_3_80')
# model.do_reinforce(scorer)
model.do_eval(training = False, filename = 'mimic_seq2seq.h256.txt', max_batch = 5000)

# model.load_params('../models/resume_seed13_100d_lr0.001_h256.model')
# ret = model.do_generate(data)
#
# from utils.eval import Evaluator
# eva = Evaluator()
# cnt = 0
# truth = []
# sum_jaccard = 0
# for line in open("seq2seq.h256.txt"):
#     if cnt % 3 == 1:
#         truth = set(line.strip().split("T: ")[1].split(" "))
#     if cnt % 3 == 2:
#         result = set(line.strip().split("Gen: ")[1].replace("END", "").strip().split(" "))
#         jaccard = eva.get_jaccard_k(truth, result)
#         sum_jaccard += jaccard
#     cnt += 1
#
# print(sum_jaccard * 3 / cnt)
#
# cnt = 0
# truth_list = []
# prediction_list = []
# for line in open("seq2seq.h256.txt"):
#     if cnt % 3 == 1:
#         truth = set(line.strip().split("T: ")[1].split(" "))
#         truth_list.append(truth)
#     if cnt % 3 == 2:
#         result = set(line.strip().split("Gen: ")[1].replace("END", "").strip().split(" "))
#         prediction_list.append(result)
#     cnt += 1
#
cnt = 0
results = []
input = []
truth = []
for line in open("mimic_seq2seq.h256.txt"):
    if cnt % 3 == 0:
        input = set(line.strip().split("S: ")[1].split(" "))
    if cnt % 3 == 1:
        if len(line.strip().split("T: ")) <= 1:
            truth = []
            continue
        truth = set(line.strip().split("T: ")[1].split(" "))
    if cnt % 3 == 2:
        result = set(line.strip().split("Gen: ")[1].replace("END", "").strip().split(" "))
        if len(truth) > 0:
            results.append((input, truth, result))
    cnt += 1
dump(results, "mimic_result_seq2seq_3_80.pkl")