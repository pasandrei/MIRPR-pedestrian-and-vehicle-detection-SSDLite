import torch.optim as optim

"""
Various optimizer setups
"""


def layer_specific_adam(model, params):
    print("AMS grad is false")
    return optim.Adam([
        {'params': model.backbone.parameters(), 'lr': params.learning_rate * params.decay_rate},
        {'params': model.loc.parameters()},
        {'params': model.conf.parameters()},
        {'params': model.additional_blocks.parameters()}
    ], lr=params.learning_rate, weight_decay=params.weight_decay, amsgrad=False)


def layer_specific_sgd(model, params):
    return optim.SGD([
        {'params': model.backbone.parameters(), 'lr': params.learning_rate * params.decay_rate},
        {'params': model.loc.parameters()},
        {'params': model.conf.parameters()},
        {'params': model.additional_blocks.parameters()}
    ], lr=params.learning_rate, weight_decay=params.weight_decay, momentum=0.9)


def plain_adam(model, params):
    return optim.Adam(model.parameters(), lr=params.learning_rate, weight_decay=params.weight_decay)


def plain_sgd(model, params):
    return optim.SGD(model.parameters(), lr=params.learning_rate,
                     weight_decay=params.weight_decay, momentum=0.9)
