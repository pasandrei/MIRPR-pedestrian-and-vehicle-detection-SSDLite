from general_config import constants, general_config


def cross_validate(model, detection_loss, valid_loader, model_evaluator, params, stats):
    """
    Goal: find the best pair of confidence threshold and NMS suppression threshold
    Args:
    - model to cross validate
    - data loader

    Return:
        - the best threshold pair
    """

    best_conf_threshold, best_suppress_threshold, best_mAP = 0, 0, 0

    conf_range = [(0.01 + i / 50) for i in range(5)]
    suppress_range = [(0.45 + i / 15) for i in range(3)]

    print(conf_range)
    print(suppress_range)
    for i in range(len(conf_range)):
        for j in range(len(suppress_range)):
            print("Current best hyperparams: ")
            print("Confidence: ", best_conf_threshold, "Suppress: ", best_suppress_threshold)

            print("Currently trying: ", conf_range[i], suppress_range[j])
            model_evaluator.output_handler.confidence_threshold = conf_range[i]
            model_evaluator.output_handler.suppress_threshold = suppress_range[j]
            cur_mAP = model_evaluator.only_mAP(model)
            print("Current mAP: ", cur_mAP)

            if cur_mAP > best_mAP:
                print("New best values found")
                best_conf_threshold, best_suppress_threshold, best_mAP = conf_range[i], suppress_range[j], cur_mAP
                params.conf_threshold = conf_range[i]
                params.suppress_threshold = suppress_range[j]
                stats.mAP = cur_mAP
                params.save(constants.params_path.format(general_config.model_id))
                stats.save(constants.stats_path.format(general_config.model_id))
                print('Params saved succesfully')
