from train.lr_policies.base_lr_handler import BaseLrHandler


class Poly_LR(BaseLrHandler):
    """
    Poly learning rate annealing: https://arxiv.org/pdf/1506.04579.pdf
    """

    def __init__(self, loader_size, optimizer, params):
        super(Poly_LR, self).__init__(optimizer, params)
        self.loader_size = loader_size
        self.batch_count = 0
        self.poly_decay = 1
        self.last_epoch = -1

    def step(self, epoch):
        if epoch != self.last_epoch:
            self.batch_count = 0
            self.last_epoch = epoch
        self.batch_count += 1
        max_iters = self.params.n_epochs * self.loader_size
        current_iters = (epoch * self.loader_size) + self.batch_count
        self.poly_decay = (1 - (current_iters / max_iters)) ** 0.9
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group['lr'] = self.param_group_lrs[i] * self.poly_decay
