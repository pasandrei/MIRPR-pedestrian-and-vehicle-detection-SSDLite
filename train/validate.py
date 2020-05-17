import torch
import datetime

from misc.model_output_handler import Model_output_handler
from utils import postprocessing, training, prints
from general_config.general_config import device


class Model_evaluator():

    def __init__(self, valid_loader, detection_loss=None, params=None, stats=None):
        """
        class used to evaluate the model on the validation set
        ARGS:
        - valid_loader - validation set dataloader
        - detection_loss - class used to compute loss
        - stats - Params object to save performance
        """
        self.valid_loader = valid_loader
        self.detection_loss = detection_loss
        self.output_handler = Model_output_handler(params)
        self.params = params
        self.stats = stats

    def complete_evaluate(self, model, optimizer, epoch=0):
        """
        evaluates model performance of the validation set, saves current model,
        optimizer stats if it is better that the best so far

        also logs info to tensorboard
        """
        print('Validation start...')
        model.eval()
        with torch.no_grad():
            losses = [0] * 4
            val_set_size = len(self.valid_loader.sampler.sampler)

            prediction_annotations = []
            prediction_id = 0

            print(datetime.datetime.now())
            for batch_idx, (input_, label, image_info) in enumerate(self.valid_loader):
                input_ = input_.to(device)
                label[0] = label[0].to(device)
                label[1] = label[1].to(device)
                output = model(input_)

                prediction_annotations, prediction_id = postprocessing.prepare_outputs_for_COCOeval(
                    output, image_info, prediction_annotations, prediction_id, self.output_handler)

                loc_loss, class_loss = self.detection_loss.ssd_loss(output, label)
                training.update_losses(losses, loc_loss.item(), class_loss.item())

                prints.print_val_batch_stats(
                    model, batch_idx, self.valid_loader, losses, self.params)

            mAP = postprocessing.evaluate_on_COCO_metrics(prediction_annotations)

            val_loss = (losses[2] + losses[3]) / val_set_size
            if self.stats.mAP < mAP:
                self.stats.mAP = mAP
                msg = 'Model saved succesfully'
                training.save_model(epoch, model, optimizer, self.params, self.stats, msg=msg)

            if self.stats.loss > val_loss:
                self.stats.loss = val_loss
                msg = 'Model saved succesfully by loss'
                training.save_model(epoch, model, optimizer, self.params,
                                    self.stats, msg=msg, by_loss=True)

            loc_loss_val, class_loss_val = losses[2] / val_set_size, losses[3] / val_set_size
            return mAP, loc_loss_val, class_loss_val

        print('Validation finished')

    def only_mAP(self, model):
        """
        only computes the mAP (for cross validation)
        """
        model.eval()
        with torch.no_grad():
            prediction_annotations = []
            prediction_id = 0
            for batch_idx, (input_, label, image_info) in enumerate(self.valid_loader):
                input_ = input_.to(device)
                output = model(input_)

                if batch_idx % 50 == 0:
                    print("Done ", batch_idx + 1, " batches")

                prediction_annotations, prediction_id = postprocessing.prepare_outputs_for_COCOeval(
                    output, image_info, prediction_annotations, prediction_id, self.output_handler)
            # map
            return postprocessing.evaluate_on_COCO_metrics(prediction_annotations)
