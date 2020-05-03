import torch
import time

from utils.postprocessing import prepare_outputs_for_COCOeval
from misc.model_output_handler import Model_output_handler
from general_config.general_config import device


class Model_evaluator():

    def __init__(self, valid_loader, detection_loss, writer=None, params=None):
        self.valid_loader = valid_loader
        self.detection_loss = detection_loss
        self.output_handler = Model_output_handler(params)
        self.writer = writer
        self.params = params

    def complete_evaluate(self, model, optimizer, train_loader, verbose, losses=[0, 0, 0, 0], epoch=0):
        model.eval()
        with torch.no_grad():
            prediction_annotations = []
            prediction_id = 0

            nr_batches = len(self.valid_loader)
            WARM_UP = 2

            counted_batches = nr_batches - WARM_UP

            average_inference_duration = average_prepare_duration = 0

            print("\n~~~~~~~~~~~~~~~~~~~~~~~~~~\n")

            for batch_idx, (input_, label, image_info) in enumerate(self.valid_loader):
                if verbose:
                    print("Batch id: ", batch_idx)
                input_ = input_.to(device)

                time_before_inference = time.time()
                output = model(input_)
                time_after_inference = time.time()

                inference_duration = time_after_inference - time_before_inference

                if verbose:
                    print("Forward propagation time: {}".format(inference_duration))

                start_duration = time.time()
                prediction_annotations, prediction_id = prepare_outputs_for_COCOeval(
                    output, image_info, prediction_annotations, prediction_id, self.output_handler)
                end_duration = time.time()

                prepare_duration = end_duration - start_duration

                if batch_idx >= WARM_UP:
                    average_inference_duration += inference_duration / counted_batches
                    average_prepare_duration += prepare_duration / counted_batches

                if verbose:
                    print("Prepare output time: ", prepare_duration, '\n')

            print("Average time to for inference (on eval): ", average_inference_duration)
            print("Total time to for inference (on eval): {} on {} batches of size {}".format(
                average_inference_duration*counted_batches, counted_batches, self.params.batch_size))
            print("-----------")
            print("Average time to prepare outputs: ", average_prepare_duration)
            print("Total time to prepare outputs: {} on {} batches of size {}".format(
                average_prepare_duration*counted_batches, counted_batches, self.params.batch_size))
