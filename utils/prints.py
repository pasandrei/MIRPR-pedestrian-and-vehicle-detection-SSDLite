def show_training_info(params):
    params_ = params.dict()
    for k, v in params_.items():
        print(str(k) + " : " + str(v))

    print("List of anchors per feature map cell: ", anchor_config.k_list)


def print_trained_parameters_count(model, optimizer):
    print('Total number of parameters of model: ', sum(p.numel() for p in model.parameters()))
    print('Total number of trainable parameters of model: ',
          sum(p.numel() for p in model.parameters() if p.requires_grad))

    print('Total number of parameters given to optimizer: ')
    print(sum(p.numel() for pg in optimizer.param_groups for p in pg['params']))
    print('Total number of trainable parameters given to optimizer: ')
    print(sum(p.numel() for pg in optimizer.param_groups for p in pg['params'] if p.requires_grad))
