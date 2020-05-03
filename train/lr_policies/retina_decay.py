class Lr_decay():
    """
    Implements learning rate decay similar to the retina net paper:
    initial learning rate of 0.01, which is then
    divided by 10 at certain steps
    """
    def __init__(self, params):
        self.params = params

    def step(self, epoch, optimizer):
        if epoch == self.params.first_decay:
            # don't want to decay backbone here as it starts at a lower lr, if it is frozen
            for idx, param_gr in enumerate(optimizer.param_groups):
                if idx == 0 and self.params.freeze_backbone:
                    continue
                param_gr['lr'] *= self.params.decay_rate

        if epoch == self.params.second_decay:
            for param_gr in optimizer.param_groups:
                param_gr['lr'] *= self.params.decay_rate
