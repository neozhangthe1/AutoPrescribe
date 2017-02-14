
# import score
#
# scorer = score.Scorer()
# print 'scorer loaded'

from models.processor import Processor
from models.coverage import CoverageModel
from exp.coverage import config_sutter as config
from utils.data import dump

config = config.get_config()
dir = 'build/'
config.saved_model_file = dir + 'rf_sutter_%s_%s_seq2seq.model' % (config.level, config.order)



print(config.saved_model_file.split('/')[-1])
p = Processor(config)
model = CoverageModel(p, config)

# model.do_train()

class Scorer(object):
    def __init__(self):
        pass

    def jaccard(self, s0, s1):
        s0 = set(s0)
        s1 = set(s1)
        intersection = len(s0.intersection(s1))
        union = len(s0.union(s1))
        score = 0.0 if union == 0 else float(intersection) / union
        # print(score)
        return score

    def predict(self, instances):
        rewards = []
        for i, instance in enumerate(instances):
            score_j = self.jaccard(instance[0], instance[1])
            rewards.append(score_j)
        return rewards


model.load_params('build/sutter_%s_%s_seq2seq.model_0_4000' % (config.level, config.order))
model.do_reinforce(Scorer())
# model.do_eval(training = False, filename = 'mimic_seq2seq.h256.txt', max_batch = 5000)

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
# cnt = 0
# results = []
# input = []
# truth = []
# for line in open("seq2seq.h256.txt"):
#     if cnt % 3 == 0:
#         input = set(line.strip().split("S: ")[1].split(" "))
#     if cnt % 3 == 1:
#         truth = set(line.strip().split("T: ")[1].split(" "))
#     if cnt % 3 == 2:
#         result = set(line.strip().split("Gen: ")[1].replace("END", "").strip().split(" "))
#         results.append((input, truth, result))
#     cnt += 1
# dump(results, "sutter_result_seq2seq_1.30.pkl")