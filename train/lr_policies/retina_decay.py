from train.lr_policies.base_lr_handler import BaseLrHandler


class Retina_decay(BaseLrHandler):
    """
    Implements learning rate decay similar to the retina net paper:
    initial learning rate which is then divided by 10 at certain steps
    """
    def __init__(self, optimizer, params):
        super(Retina_decay, self).__init__(optimizer, params)

    def step(self, epoch):
        if epoch == self.params.first_decay:
            for idx, param_gr in enumerate(self.optimizer.param_groups):
                param_gr['lr'] = self.param_group_lrs[idx] * self.params.decay_rate

        if epoch == self.params.second_decay:
            for idx, param_gr in enumerate(self.optimizer.param_groups):
                param_gr['lr'] = self.param_group_lrs[idx] * (self.params.decay_rate ** 2)
