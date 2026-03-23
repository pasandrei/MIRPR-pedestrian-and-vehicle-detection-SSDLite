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


def benchmark_num_workers(model, optimizer, detection_loss, params, scaler, candidates=[8, 16], warmup=20, count=200):
    """Benchmarks different num_workers values and sets the fastest one."""
    import time
    from data import dataloaders as dl

    best_time, best_workers = float('inf'), candidates[0]
    for nw in candidates:
        general_config.num_workers = nw
        loader, _ = dl.get_dataloaders(params)
        model.train()
        for i, (input_, label, _) in enumerate(loader):
            if i >= warmup + count:
                break
            input_ = input_.to(general_config.device)
            label[0] = label[0].to(general_config.device)
            label[1] = label[1].to(general_config.device)
            optimizer.zero_grad()
            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    output = model(input_)
                    l_loss, c_loss = detection_loss.ssd_loss(output, label)
                    loss = l_loss + c_loss
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                output = model(input_)
                l_loss, c_loss = detection_loss.ssd_loss(output, label)
                loss = l_loss + c_loss
                loss.backward()
                optimizer.step()
            if i == warmup - 1:
                torch.cuda.synchronize()
                start = time.time()
        torch.cuda.synchronize()
        elapsed = time.time() - start
        print(f"num_workers={nw}: {count} batches in {elapsed:.2f}s ({elapsed/count*1000:.1f}ms/batch)")
        if elapsed < best_time:
            best_time, best_workers = elapsed, nw

    print(f"Selected num_workers={best_workers}")
    general_config.num_workers = best_workers


def run(train_model=True, load_checkpoint=False, cross_validate=False,
        validate=False, mixed_precision=False, test_dev=False,
        auto_workers=False, run_name=None):
    """
    Arguments:
    train_model - train model
    load_checkpoint - load a pretrained model
    validate - run evaluation
    cross_validate - cross validate for best nms thresold and positive confidence
    mixed_precision - use mixed_precision training
    test_dev - run model on coco test-dev set
    auto_workers - benchmark num_workers 8 vs 16 and pick the fastest
    """
    torch.manual_seed(2)
    random.seed(2)

    params = Params(constants.params_path.format(general_config.model_id))
    stats = Params(constants.stats_path.format(general_config.model_id))
    prints.show_training_info(params)

    model = training.model_setup(params)
    model = torch.compile(model)
    optimizer = training.optimizer_setup(model, params)

    scaler = torch.amp.GradScaler('cuda') if mixed_precision else None

    if auto_workers:
        detection_loss = Detection_Loss(params)
        benchmark_num_workers(model, optimizer, detection_loss, params, scaler)
        # reinitialize model and optimizer after benchmark
        model = training.model_setup(params)
        model = torch.compile(model)
        optimizer = training.optimizer_setup(model, params)
        scaler = torch.amp.GradScaler('cuda') if mixed_precision else None

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
    if run_name:
        writer = SummaryWriter(comment=f"_{run_name}")
    else:
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
                    scaler)


if __name__ == '__main__':
    run(mixed_precision=True)
