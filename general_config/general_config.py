import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_stats_step = 10
eval_step = 4
agnostic_nms = True