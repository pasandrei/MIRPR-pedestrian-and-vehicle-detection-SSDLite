from torch.utils.tensorboard import SummaryWriter
from train.loss_fn import Detection_Loss
import torch
import random

from train import train
from train.params import Params
from train.validate import Model_evaluator
from misc import cross_validation
from general_config import constants, general_config
from data import dataloaders
from custom_inference.run import Custom_Infernce

from utils import prints
from utils import training

try:
    from apex import amp
    APEX_AVAILABLE = True
except ImportError:
    APEX_AVAILABLE = False
    raise ImportError("Please install APEX from https://github.com/nvidia/apex")


def run(train_model=True, load_checkpoint=False, cross_validate=False,
        validate=False, mixed_precision=False, test_dev=False):
    """
    Arguments:
    train_model - train model
    load_checkpoint - load a pretrained model
    validate - run evaluation
    cross_validate - cross validate for best nms thresold and positive confidence
    mixed_precision - use mixed_precision training
    test_dev - run model on coco test-dev set
    """
    torch.manual_seed(2)
    random.seed(2)

    params = Params(constants.params_path.format(general_config.model_id))
    stats = Params(constants.stats_path.format(general_config.model_id))
    prints.show_training_info(params)

    model = training.model_setup(params)
    optimizer = training.optimizer_setup(model, params)

    if APEX_AVAILABLE and mixed_precision:
        model, optimizer = amp.initialize(
            model, optimizer, opt_level="O2"
        )

    start_epoch = 0
    if load_checkpoint:
        model, optimizer, start_epoch = training.load_model(model, params, optimizer)
    prints.print_trained_parameters_count(model, optimizer)

    if test_dev:
        print("Running evaluation on test-dev")
        test_loader = dataloaders.get_test_dev(params)
        prints.print_dataset_stats(valid_loader=test_loader)
        test_model_evaluator = Model_evaluator(test_loader, params=params)
        test_model_evaluator.only_mAP(model)
        return

    # tensorboard
    writer = SummaryWriter(filename_suffix=general_config.model_id)

    if train_model:
        train_loader, valid_loader = training.prepare_datasets(params)
        prints.print_dataset_stats(train_loader, valid_loader)
    else:
        valid_loader = dataloaders.get_dataloaders_test(params)

    detection_loss = Detection_Loss(params)
    model_evaluator = Model_evaluator(valid_loader, detection_loss,
                                      params=params, stats=stats)
    if train_model:
        lr_decay_policy = training.lr_decay_policy_setup(params, optimizer, len(train_loader))

    if validate:
        print("Checkpoint epoch: ", start_epoch)
        prints.print_dataset_stats(valid_loader=valid_loader)
        model_evaluator.complete_evaluate(model, optimizer)

    if cross_validate:
        cross_validation.cross_validate(
            model, detection_loss, valid_loader, model_evaluator, params, stats)

    if train_model:
        train.train(model, optimizer, train_loader, model_evaluator,
                    detection_loss, params, writer, lr_decay_policy, start_epoch,
                    APEX_AVAILABLE and mixed_precision)


if __name__ == '__main__':
    run()
