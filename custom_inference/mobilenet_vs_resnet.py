"""
Simple comparison of cpu vs gpu performance of MobileNetV2 vs different ResNets
"""
import time
import torchvision.models as models
import torch


def compare(device, height, width):
    print("Running on: ", device)
    resnet18 = models.resnet18(pretrained=True).to(device)
    resnet50 = models.resnet50(pretrained=True).to(device)
    resnet101 = models.resnet101(pretrained=True).to(device)
    mobilenet = models.mobilenet_v2(pretrained=True).to(device)

    dummy_input = torch.randn(1, 3, height, width).to(device)

    print("Times taken: \n")

    resnet18_time = run_model(dummy_input, resnet18)
    print("Resnet18: ", "{:.4f}".format(resnet18_time))

    resnet50_time = run_model(dummy_input, resnet50)
    print("Resnet50: ", "{:.4f}".format(resnet50_time))

    resnet101_time = run_model(dummy_input, resnet101)
    print("Resnet101: ", "{:.4f}".format(resnet101_time))

    mobilenet_time = run_model(dummy_input, mobilenet)
    print("Mobilenet: ", "{:.4f}".format(mobilenet_time))


def run_model(input_, model):
    """
    average runtime of 10 runs
    """
    model.eval()
    run = 0
    total = 0
    with torch.no_grad():
        while run < 10:
            start = time.time()
            _ = model(input_)
            taken = time.time() - start
            total += taken
            run += 1
        return total / 10
