
# import score
#
# scorer = score.Scorer()
# print 'scorer loaded'

from models.processor import Processor
from models.leap import LEAPModel
from exp.coverage import config_mimic as config
from utils.data import dump

config = config.get_config()
dir = 'build/'
config.saved_model_file = dir + 'mimic_%s_%s_seq2seq.model' % (config.level, config.order)

print(config.saved_model_file.split('/')[-1])
p = Processor(config)
model = LEAPModel(p, config)

model.do_train()

