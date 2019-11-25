def lr_decay(optimizer, decay_rate=0.97):
    '''
    Implements learning rate decay as described in the mobileNetV2 paper:
    - after each epoch simply multiply the lr by 0.97
    '''

    for param_gr in optimizer.param_groups:
        param_gr['lr'] *= decay_rate
