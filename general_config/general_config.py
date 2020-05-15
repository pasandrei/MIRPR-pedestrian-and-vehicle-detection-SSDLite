import torch
from general_config import constants

model_id = constants.ssd_modified
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = "cpu"
batch_stats_step = 10
eval_step = 1
agnostic_nms = True
num_workers = 0
