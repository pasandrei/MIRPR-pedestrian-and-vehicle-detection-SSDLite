class BaseLrHandler():
    def __init__(self, optimizer, params):
        self.params = params
        self.optimizer = optimizer

        # get initial lr for each param group
        self.param_group_lrs = []
        for param_group in self.optimizer.param_groups:
            self.param_group_lrs.append(param_group['lr'])

    def warm_up(self, batch_idx, train_size):
        """
        linearly increase learning_rate 10x during the first epoch
        """
        batch_idx += 1
        for i, param_group in enumerate(self.optimizer.param_groups):
            progress = (batch_idx / train_size)
            lr = self.param_group_lrs[i] / 10 + progress * self.param_group_lrs[i] * 0.9
            param_group['lr'] = lr
