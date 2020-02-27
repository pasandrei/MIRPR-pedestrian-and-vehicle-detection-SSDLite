import torch

from architectures.models import SSDLite
from train.params import Params
from general_config import path_config
from train.validate import Model_evaluator
from benchmarks import train_benchmark, inference_benchmark
from data import dataloaders
from train.optimizer_handler import plain_adam
from train.loss_fn import Detection_Loss
from utils.training import model_setup


def run_training(model_id="ssdlite", benchmark_train=False, benchmark_inference=False, verbose=False):
    params = Params(path_config.params_path.format(model_id))

    model = model_setup(params)

    optimizer = plain_adam(model, params)

    train_loader, valid_loader = dataloaders.get_dataloaders(params)

    detection_loss = Detection_Loss(params)

    if benchmark_train:
        model_evaluator = None
        train_benchmark.train(model, optimizer, train_loader, model_evaluator,
                              detection_loss, params, verbose)
    if benchmark_inference:
        model_evaluator = inference_benchmark.Model_evaluator(
            valid_loader, detection_loss, writer=None, params=params)
        model_evaluator.complete_evaluate(model, optimizer, train_loader, verbose)

# run_training()
