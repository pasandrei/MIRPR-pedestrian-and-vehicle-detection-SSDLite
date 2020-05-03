class Poly_LR():
    """
    Poly learning rate annealing: https://arxiv.org/pdf/1506.04579.pdf
    """

    def __init__(self, loader_size, params):
        self.params = params
        self.loader_size = loader_size
        self.batch_count = 0
        self.poly_decay = 1
        self.last_epoch = -1

    def step(self, epoch, optimizer):
        if epoch != self.last_epoch:
            self.batch_count = 0
            self.last_epoch = epoch
        self.batch_count += 1
        max_iters = self.params.n_epochs * self.loader_size
        current_iters = (epoch * self.loader_size) + self.batch_count
        self.poly_decay = (1 - (current_iters / max_iters)) ** 0.9
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.params.learning_rate * self.poly_decay
