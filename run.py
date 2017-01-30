
# import score
#
# scorer = score.Scorer()
# print 'scorer loaded'

from models.processor import Processor
from models.coverage import CoverageModel
from exp.coverage import config1 as config

config = config.get_config()

print(config.saved_model_file.split('/')[-1])
p = Processor(config)
model = CoverageModel(p, config)

model.do_train()

# model.load_params('../models/split_seed13_100d_lr0.001_h256.model')
# model.do_reinforce(scorer)
# model.do_eval(training = False, filename = 'after.reinforce.h256.txt', max_batch = 5)

# model.load_params('../models/resume_seed13_100d_lr0.001_h256.model')
# data = [[u"出售哈士奇", u"随便写点什么"]]
# ret = model.do_generate(data)
