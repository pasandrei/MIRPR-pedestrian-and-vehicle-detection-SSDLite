class Lr_decay:
    '''
    Implements learning rate decay as described in the retina net paper:
    initial learning rate of 0.01, which is then
    divided by 10 at 60k and again at 80k iterations
    '''
    def __init__(self, lr):
        self.lr = lr
        self.current_step = 0

    def step(self, optimizer):
        self.current_step += 1
        if self.current_step == 60000:
            for param_gr in optimizer.param_groups:
                param_gr['lr'] /= 10
        if self.current_step == 80000:
            for param_gr in optimizer.param_groups:
                param_gr['lr'] /= 10
