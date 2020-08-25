class BaseLrHandler():
    def __init__(self, optimizer, params):
        self.params = params
        self.optimizer = optimizer
        self.batch_num = 0

        # get initial lr for each param group
        self.param_group_lrs = []
        for param_group in self.optimizer.param_groups:
            self.param_group_lrs.append(param_group['lr'])

    def warm_up(self, train_size):
        """
        linearly increase learning_rate 100x during the first epoch
        """
        self.batch_num += 1
        for i, param_group in enumerate(self.optimizer.param_groups):
            progress = (self.batch_num / (train_size * self.params.warm_up))
            lr = self.param_group_lrs[i] * 0.001 + progress * self.param_group_lrs[i] * 0.999
            param_group['lr'] = lr
