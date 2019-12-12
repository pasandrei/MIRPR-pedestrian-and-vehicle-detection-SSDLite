import torch
import datetime
import json

from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
from misc.model_output_handler import *
from misc.utils import *


class Model_evaluator():

    def __init__(self, valid_loader, detection_loss, conf_thresh=0.35, suppress_thresh=0.5, writer=None, params=None):
        self.valid_loader = valid_loader
        self.detection_loss = detection_loss
        self.output_handler = Model_output_handler(
            self.detection_loss.device, conf_threshold=conf_thresh, suppress_threshold=suppress_thresh)
        self.writer = writer
        self.params = params

    def complete_evaluate(self, model, optimizer, train_loader, losses, epoch):
        '''
        evaluates model performance of the validation set, saves current model, optimizer, stats if it is better that the best so far
        also logs info to tensorboard
        '''

        eval_step_avg_factor = self.params.eval_step * len(train_loader.sampler.sampler)
        loc_loss_train, class_loss_train = losses[2] / \
            eval_step_avg_factor, losses[3] / eval_step_avg_factor

        print('Average train loss at eval start: Localization: {}; Classification: {}'.format(
            loc_loss_train, class_loss_train))

        print('Validation start...')
        model.eval()
        with torch.no_grad():
            loc_loss_val, class_loss_val = 0, 0
            val_set_size = len(self.valid_loader.sampler.sampler)
            one_tenth_of_loader = len(self.valid_loader) // 10

            prediction_annotations = []
            prediction_id = 0

            for batch_idx, (input_, label, image_info) in enumerate(self.valid_loader):
                # print(datetime.datetime.now())
                input_ = input_.to(self.detection_loss.device)
                output = model(input_)

                prediction_annotations, prediction_id = prepare_outputs_for_COCOeval(
                    output, image_info, prediction_annotations, prediction_id, self.output_handler)

                loc_loss, class_loss = self.detection_loss.ssd_loss(output, label)
                loc_loss_val += loc_loss.item()
                class_loss_val += class_loss.item()

                if batch_idx % one_tenth_of_loader == 0 and batch_idx > 0:
                    # nr_images = (batch_idx + 1) * self.params.batch_size
                    nr_images = 100
                    print(datetime.datetime.now())
                    print("Average Loc Loss: ", loc_loss_val /
                          nr_images)
                    print("Average Class Loss: ", class_loss_val /
                          nr_images, " until batch: ", batch_idx)
                    loc_loss_val, class_loss_val = 0, 0

            SAVE_PATH = 'misc/experiments/{}/model_checkpoint'.format(self.params.model_id)

            mAP = evaluate_on_COCO_metrics(prediction_annotations)

            val_loss = (class_loss_val + loc_loss_val) / val_set_size
            if self.params.mAP < mAP:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, SAVE_PATH)
                self.params.loss = val_loss
                self.params.mAP = mAP
                self.params.save('misc/experiments/ssdnet/params.json')
                print('Model saved succesfully')

            # tensorboard
            average_precision = 0
            update_tensorboard_graphs(self.writer, loc_loss_train, class_loss_train,
                                      loc_loss_val, class_loss_val, average_precision, epoch)

        print('Validation finished')

    def only_mAP(self, model):
        """
        only computes the mAP (for cross validation)
        """
        model.eval()
        with torch.no_grad():
            loc_loss_val, class_loss_val = 0, 0
            one_tenth_of_loader = len(self.valid_loader) // 10

            prediction_annotations = []
            prediction_id = 0
            for batch_idx, (input_, label, image_info) in enumerate(self.valid_loader):
                # print(datetime.datetime.now())
                input_ = input_.to(self.detection_loss.device)
                output = model(input_)

                prediction_annotations, prediction_id = prepare_outputs_for_COCOeval(
                    output, image_info, prediction_annotations, prediction_id, self.output_handler)

                loc_loss, class_loss = self.detection_loss.ssd_loss(output, label)
                loc_loss_val += loc_loss.item()
                class_loss_val += class_loss.item()

                if batch_idx % one_tenth_of_loader == 0 and batch_idx > 0:
                    # nr_images = (batch_idx + 1) * self.params.batch_size
                    nr_images = 100
                    print(datetime.datetime.now())
                    print("Average Loc Loss: ", loc_loss_val /
                          nr_images)
                    print("Average Class Loss: ", class_loss_val /
                          nr_images, " until batch: ", batch_idx)

                    loc_loss_val, class_loss_val = 0, 0

            # map
            return evaluate_on_COCO_metrics(prediction_annotations)
