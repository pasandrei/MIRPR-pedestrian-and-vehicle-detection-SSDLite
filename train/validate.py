import torch
import datetime

from misc.model_output_handler import Model_output_handler
from utils.postprocessing import *
from utils.training import *
from general_config.config import device


class Model_evaluator():

    def __init__(self, valid_loader, detection_loss, writer=None, params=None, stats=None):
        self.valid_loader = valid_loader
        self.detection_loss = detection_loss
        self.output_handler = Model_output_handler(params)
        self.writer = writer
        self.params = params
        self.stats = stats

    def complete_evaluate(self, model, optimizer, train_loader, losses=[0, 0, 0, 0], epoch=0):
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
            one_tenth_of_loader = len(self.valid_loader) // self.params.train_stats_step

            prediction_annotations = []
            prediction_id = 0

            print(datetime.datetime.now())
            for batch_idx, (input_, label, image_info) in enumerate(self.valid_loader):
                input_ = input_.to(device)
                output = model(input_)

                prediction_annotations, prediction_id = prepare_outputs_for_COCOeval(
                    output, image_info, prediction_annotations, prediction_id, self.output_handler)

                loc_loss, class_loss = self.detection_loss.ssd_loss(output, label)
                loc_loss_val += loc_loss.item()
                class_loss_val += class_loss.item()

                if (batch_idx + 1) % one_tenth_of_loader == 0:
                    print(datetime.datetime.now())
                    nr_images = (batch_idx + 1) * self.params.batch_size
                    print("Average Loc Loss: ", loc_loss_val / nr_images)
                    print("Average Class Loss: ", class_loss_val / nr_images,
                          " until batch: ", batch_idx)

            mAP = evaluate_on_COCO_metrics(prediction_annotations)

            val_loss = (class_loss_val + loc_loss_val) / val_set_size
            if self.stats.mAP < mAP:
                self.stats.mAP = mAP
                msg = 'Model saved succesfully'
                save_model(epoch, model, optimizer, self.params, self.stats, msg=msg)

            if self.stats.loss > val_loss:
                self.stats.loss = val_loss
                msg = 'Model saved succesfully by loss'
                save_model(epoch, model, optimizer, self.params, self.stats, msg=msg, by_loss=True)

            # tensorboard
            loc_loss_val, class_loss_val = loc_loss_val / val_set_size, class_loss_val / val_set_size
            update_tensorboard_graphs(self.writer, loc_loss_train, class_loss_train,
                                      loc_loss_val, class_loss_val, mAP, epoch)

        print('Validation finished')

    def only_mAP(self, model):
        """
        only computes the mAP (for cross validation)
        """
        model.eval()
        with torch.no_grad():
            loc_loss_val, class_loss_val = 0, 0
            one_tenth_of_loader = len(self.valid_loader) // self.params.train_stats_step

            prediction_annotations = []
            prediction_id = 0
            for batch_idx, (input_, label, image_info) in enumerate(self.valid_loader):
                input_ = input_.to(device)
                output = model(input_)

                prediction_annotations, prediction_id = prepare_outputs_for_COCOeval(
                    output, image_info, prediction_annotations, prediction_id, self.output_handler)

                loc_loss, class_loss = self.detection_loss.ssd_loss(output, label)
                loc_loss_val += loc_loss.item()
                class_loss_val += class_loss.item()

                if (batch_idx + 1) % one_tenth_of_loader == 0:
                    print(datetime.datetime.now())
                    nr_images = (batch_idx + 1) * self.params.batch_size
                    print("Average Loc Loss: ", loc_loss_val /
                          nr_images)
                    print("Average Class Loss: ", class_loss_val /
                          nr_images, " until batch: ", batch_idx)

            # map
            return evaluate_on_COCO_metrics(prediction_annotations)
