
# import score
#
# scorer = score.Scorer()
# print 'scorer loaded'

from models.processor import Processor
from models.coverage import CoverageModel
from exp.coverage import config_mimic as config
from utils.data import dump

config = config.get_config()

print(config.saved_model_file.split('/')[-1])
p = Processor(config)
model = CoverageModel(p, config)

model.do_train()

