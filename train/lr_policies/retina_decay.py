class Lr_decay:
    '''
    Implements learning rate decay similar to the retina net paper:
    initial learning rate of 0.01, which is then
    divided by 10 at 20 epochs and again at 35 epochs
    '''

    def __init__(self, lr):
        self.lr = lr
        self.current_step = 0

    def step(self, optimizer):
        self.current_step += 1
        if self.current_step == 20:
            for param_gr in optimizer.param_groups:
                param_gr['lr'] /= 10
        if self.current_step == 35:
            for param_gr in optimizer.param_groups:
                param_gr['lr'] /= 10
