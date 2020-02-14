import torch

from architectures.models import SSDNet
from train.config import Params
from general_config import anchor_config, path_config
from train.validate import Model_evaluator
from benchmarks import train_benchmark, inference_benchmark
from data import dataloaders
from train.optimizer_handler import plain_adam
from train.helpers import *
from train.loss_fn import Detection_Loss


def run_training(benchmark_on_train=False, benchmark_on_inference=False):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    params = Params(path_config.params_path)

    model = SSDNet.SSD_Head(n_classes=params.n_classes, k_list=anchor_config.k_list)
    model.to(device)

    optimizer = plain_adam(model, params)

    train_loader, valid_loader = dataloaders.get_dataloaders(params)

    anchors, grid_sizes = create_anchors()
    anchors, grid_sizes = anchors.to(device), grid_sizes.to(device)

    detection_loss = Detection_Loss(anchors, grid_sizes, device, params, focal_loss=True,
                                    hard_negative=False)

    if benchmark_on_train:
        model_evaluator = None
        train_benchmark.train(model, optimizer, train_loader, model_evaluator,
                              detection_loss, params)
    if benchmark_on_inference:
        model_evaluator = inference_benchmark.Model_evaluator(
            valid_loader, detection_loss, writer=None, params=params)
        model_evaluator.complete_evaluate(model, optimizer, train_loader)

# run_training()
