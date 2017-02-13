
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
config.saved_model_file = dir + 'sutter_clean_%s_sorted_seq2seq.model' % config.level


print(config.saved_model_file.split('/')[-1])
p = Processor(config)
model = CoverageModel(p, config)

model.do_train()

