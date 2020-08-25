import torch
from general_config import constants

model_id = constants.ssdlite
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = "cpu"
batch_stats_step = 10
eval_step = 2
agnostic_nms = False
num_workers = 4
