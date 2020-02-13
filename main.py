from torch.utils.tensorboard import SummaryWriter
from train.loss_fn import Detection_Loss
import torch

from train import train
from train.config import Params
from train.validate import Model_evaluator
from misc import cross_validation
from misc.model_output_handler import Model_output_handler
from jaad_data import inference
from general_config import path_config

from utils.prints import *
from utils.training import *


def run(train_model=False, load_model=False, eval_only=False, cross_validate=False, jaad=False):
    """
    Arguments:
    train_model - training
    load_model - load a pretrained model
    eval_only - only inference
    cross_validate - cross validate for best nms thresold and positive confidence
    jaad - inference on jaad videos
    """
    torch.manual_seed(2)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    params = Params(path_config.params_path)

    train_loader, valid_loader = prepare_datasets(params)
    model, optimizer = model_optimizer_setup(device, params)

    if jaad:
        model, _, _ = load_model(model, optimizer, params)
        handler = Model_output_handler(params)
        inference.jaad_inference(model, handler)

    # tensorboard,
    writer = SummaryWriter(filename_suffix=params.model_id)

    detection_loss = Detection_Loss(device, params)
    model_evaluator = Model_evaluator(valid_loader, detection_loss, writer=writer, params=params)

    start_epoch = 0
    if load_model:
        model, optimizer, start_epoch = load_model(model, optimizer, params)

    if eval_only:
        model_evaluator.complete_evaluate(model, optimizer, train_loader)

    if cross_validate:
        cross_validation.cross_validate(
            model, detection_loss, valid_loader, model_evaluator, params)

    if train_model:
        train.train(model, optimizer, train_loader, model_evaluator,
                    detection_loss, params, start_epoch)
