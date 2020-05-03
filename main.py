from torch.utils.tensorboard import SummaryWriter
from train.loss_fn import Detection_Loss
import torch
import random

from train import train
from train.params import Params
from train.validate import Model_evaluator
from misc import cross_validation
from misc.model_output_handler import Model_output_handler
from jaad_data import inference
from general_config import constants

from utils import prints
from utils import training

try:
    from apex import amp
    APEX_AVAILABLE = True
except ImportError:
    APEX_AVAILABLE = False
    raise ImportError("Please install APEX from https://github.com/nvidia/apex")


def run(model_id="ssdlite", train_model=True, load_checkpoint=False, cross_validate=False,
        validate=False, jaad=False, mixed_precision=False):
    """
    Arguments:
    model_id - id of the model to be trained
    train_model - training
    load_checkpoint - load a pretrained model
    validate - run evaluation
    cross_validate - cross validate for best nms thresold and positive confidence
    jaad - inference on jaad videos
    """
    torch.manual_seed(2)
    random.seed(2)

    params = Params(constants.params_path.format(model_id))
    stats = Params(constants.stats_path.format(model_id))
    prints.show_training_info(params)

    train_loader, valid_loader = training.prepare_datasets(params)
    prints.print_dataset_stats(train_loader, valid_loader)

    model = training.model_setup(params)
    optimizer = training.optimizer_setup(model, params)

    if APEX_AVAILABLE and mixed_precision:
        model, optimizer = amp.initialize(
            model, optimizer, opt_level="O2"
        )

    if jaad:
        model, _, _ = training.load_model(model, params, optimizer)
        handler = Model_output_handler(params)
        inference.jaad_inference(model, handler)

    # tensorboard
    writer = SummaryWriter(filename_suffix=params.model_id)

    detection_loss = Detection_Loss(params)
    model_evaluator = Model_evaluator(valid_loader, detection_loss,
                                      params=params, stats=stats)

    start_epoch = 0
    if load_checkpoint:
        model, optimizer, start_epoch = training.load_model(model, params, optimizer)
    prints.print_trained_parameters_count(model, optimizer)

    if validate:
        model_evaluator.complete_evaluate(model, optimizer)

    if cross_validate:
        cross_validation.cross_validate(
            model, detection_loss, valid_loader, model_evaluator, params)

    if train_model:
        train.train(model, optimizer, train_loader, model_evaluator,
                    detection_loss, params, writer, start_epoch,
                    APEX_AVAILABLE and mixed_precision)


if __name__ == '__main__':
    run()
