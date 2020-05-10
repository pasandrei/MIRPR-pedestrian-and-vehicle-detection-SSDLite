import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = "cpu"
batch_stats_step = 10
eval_step = 1
agnostic_nms = False
