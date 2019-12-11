from misc.model_output_handler import *
from train import validate


def cross_validate(model, valid_loader, device, params):
    """
    Goal: find the best pair of confidence threshold and NMS suppression threshold
    Args:
    - model to cross validate
    - data loader

    Return:
        - the best threshold pair
    """

    best_conf_threshold, best_suppress_threshold, best_mAP = 0, 0, 0

    conf_range = [(0.2 + i / 100) for i in range(26)]
    suppress_range = [(0.4 + i / 50) for i in range(11)]

    for i in range(len(conf_range)):
        for j in range(len(suppress_range)):
            print("Current best hyperparams: ")
            print("Confidence: ", best_conf_threshold, "Suppress: ", best_suppress_threshold)

            cur_conf_threshold, cur_suppress_threshold = conf_range[i], suppress_range[j]
            output_handler = Model_output_handler(
                device, cur_conf_threshold, cur_suppress_threshold)

            cur_mAP = validate.evaluate(model, valid_loader, device, optimizer=None, conf_threshold=cur_conf_threshold,
                                        suppress_threshold=cur_suppress_threshold, cross_validate=True, params=params)

            if cur_mAP > best_mAP:
                print("New best values found")
                best_conf_threshold, best_suppress_threshold, best_mAP = cur_conf_threshold, cur_suppress_threshold, cur_mAP
                params.conf_threshold = cur_conf_threshold
                params.suppress_threshold = cur_suppress_threshold
                params.save('misc/experiments/ssdnet/params.json')
                print('Params saved succesfully')
