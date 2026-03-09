import math
from train.lr_policies.base_lr_handler import BaseLrHandler


class Cosine_LR(BaseLrHandler):
    """
    Cosine annealing LR schedule: decays LR from initial value to 0
    following a cosine curve over total epochs.
    """
    def __init__(self, optimizer, params):
        super(Cosine_LR, self).__init__(optimizer, params)

    def step(self, epoch):
        progress = epoch / self.params.n_epochs
        scale = 0.5 * (1 + math.cos(math.pi * progress))
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group['lr'] = self.param_group_lrs[i] * scale
