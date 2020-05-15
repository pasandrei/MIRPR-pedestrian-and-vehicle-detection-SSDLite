import time
import torch

from train.params import Params
from misc.model_output_handler import Model_output_handler
from general_config import constants, general_config
from utils import training
from utils.box_computations import wh2corners_numpy
from utils.postprocessing import nms, postprocess_until_nms
from data import dataloaders


class Speed_testing():
    def __init__(self, runs=100, print_each_run=False, device="cpu"):
        self.runs = runs
        self.device = device
        self.params = Params(constants.params_path.format(general_config.model_id))

        self.model = training.model_setup(self.params)
        self.model = training.load_weigths_only(self.model, self.params)
        self.model = self.model.to(self.device)
        self.model.eval()

        self.output_handler = Model_output_handler(self.params)
        self.print_each_run = print_each_run

        self.params.batch_size = 1
        self.valid_loader = dataloaders.get_dataloaders_test(self.params)

    def speed_test(self, custom_settings=None):
        """
        custom_settings, if set, should be a tuple of (nms_threshold, conf_threshold, device), this
        device - cuda:0 or cpu
        !!! overwrites the original settings
        """
        self.valid_loader_iter = iter(self.valid_loader)
        if custom_settings:
            print("Current custom settings: ", custom_settings)
            nms_thresh, conf_thresh, device = custom_settings
            self.output_handler.suppress_threshold = nms_thresh
            self.output_handler.confidence_threshold = conf_thresh
            self.device = device
            self.model.to(self.device)
        run = 0
        times_model, times_pre_nms, times_nms, in_nms_boxes = [], [], [], 0
        while run < self.runs:
            (boxes, confs), image_info, last_model = self.val_image_output()
            times_model.append(last_model)

            start_pre_nms = time.time()
            boxes, classes = postprocess_until_nms(self.output_handler, boxes,
                                                   confs, image_info[0][1])
            last_pre_nms = time.time() - start_pre_nms
            times_pre_nms.append(last_pre_nms)

            boxes = wh2corners_numpy(boxes[:, :2], boxes[:, 2:])
            start_nms = time.time()
            boxes = boxes[:200]
            in_nms_boxes += len(boxes)
            _ = nms(boxes, classes, self.output_handler.suppress_threshold)
            last_nms = time.time() - start_nms
            times_nms.append(last_nms)
            if self.print_each_run:
                print("Time for last run\n")
                print("Model: ", last_model)
                print("Pre nms: ", last_pre_nms)
                print("NMS: ", last_nms, "\n")
            run += 1

        print("Done ", self.runs, " runs")
        print("\n\n\n")

        total_model, total_pre_nms, total_nms = sum(times_model), sum(times_pre_nms), sum(times_nms)
        print("Total time of model: ", total_model)
        print("Mean time model: ", total_model / len(times_model))
        print("Total time of pre nms: ", total_pre_nms)
        print("Mean time pre nms: ", total_pre_nms / len(times_pre_nms))
        print("Total time of nms: ", total_nms)
        print("Mean time nms: ", total_nms / len(times_nms))
        print("Mean number of boxes processed by nms: ", "{:.2f}".format(in_nms_boxes / len(times_nms)))

        print("\n\n\n")

        total_time = sum(times_model + times_pre_nms + times_nms)
        print("Total time taken: ", total_time)
        print("Percentage of model: ", "{:.2f}".format(total_model / total_time * 100))
        print("Percentage of pre nms: ", "{:.2f}".format(total_pre_nms / total_time * 100))
        print("Percentage of nms: ", "{:.2f}".format(total_nms / total_time * 100))

    def val_image_output(self):
        with torch.no_grad():
            input_, _, image_info = next(self.valid_loader_iter)
            start = time.time()
            input_ = input_.to(self.device)
            boxes, confs = self.model(input_)
            boxes = boxes.squeeze().permute(1, 0)
            confs = confs.squeeze().permute(1, 0)
            last_model = time.time() - start
            return (boxes, confs), image_info, last_model
