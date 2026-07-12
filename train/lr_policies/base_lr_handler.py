class BaseLrHandler():
    def __init__(self, optimizer, params):
        self.params = params
        self.optimizer = optimizer

        # get initial lr for each param group
        self.param_group_lrs = []
        for param_group in self.optimizer.param_groups:
            self.param_group_lrs.append(param_group['lr'])

    def warm_up(self, epoch, batch_idx, train_size, warmup_epochs=1):
        """
        linearly increase learning_rate from lr/100 to lr over warmup_epochs
        """
        batch_idx += 1
        progress = (epoch * train_size + batch_idx) / (warmup_epochs * train_size)
        for i, param_group in enumerate(self.optimizer.param_groups):
            lr = self.param_group_lrs[i] / 100 + progress * self.param_group_lrs[i] * 0.99
            param_group['lr'] = lr
