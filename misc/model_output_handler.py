from train.helpers import *
from misc.postprocessing import *
from my_tests import anchor_mapping

import numpy as np


class Model_output_handler():

    def __init__(self, device):
        self.device = device
        self.unnorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        self.confidence_threshold = 0.25
        anchors, grid_sizes = create_anchors()
        self.anchors_hw, self.grid_sizes = anchors.to(self.device), grid_sizes.to(self.device)
        self.corner_anchors = hw2corners(anchors[:, :2], anchors[:, 2:])

    def process_outputs(self, bbox_predictions, classification_predictions, image_info):
        """
        returns complete model outputs in format array of:
        bbox, class id, confidence
        """
        prediction_bboxes, predicted_classes, highest_confidence_for_predictions, _ = self.__get_sorted_predictions(
            bbox_predictions, classification_predictions, image_info)

        indeces_kept_by_nms = nms(prediction_bboxes)

        # new structure: array of bbox, class, confidence

        complete_outputs = np.concatenate(
            (prediction_bboxes, predicted_classes, highest_confidence_for_predictions), axis=1)

        return complete_outputs[indeces_kept_by_nms]

    def test_anchor_mapping(self, image, bbox_predictions, classification_predictions, gt_bbox, gt_class, image_info):
        """
        Args: all input is required per image

        computes:
            - image upscaled and unnormalized as numpy array
            - predicted bboxes higher than a threshold, sorted by predicted confidence, at right scale
            - gt bboxes for the image, at right scale
        """
        overlaps = jaccard(gt_bbox, self.corner_anchors)

        prediction_bboxes, predicted_classes, highest_confidence_for_predictions, high_confidence_indeces = self.__get_sorted_predictions(
            bbox_predictions, classification_predictions, image_info)

        # map each anchor to the highest IOU obj, gt_idx - ids of mapped objects
        _, _, pos_idx = map_to_ground_truth(
            overlaps, gt_bbox, gt_class)

        indeces_kept_by_nms = nms(prediction_bboxes)

        image = self.__unnorm_scale_image(image)
        pos_idx = (pos_idx.cpu().numpy())
        gt_bbox = self.__rescale_bboxes(gt_bbox, image_info[1])
        bbox_predictions = self.__convert_bboxes_to_workable_data(bbox_predictions, image_info[1])
        anchor_mapping.test(bbox_predictions, image, self.__rescale_bboxes(self.corner_anchors, image_info[1]),
                            prediction_bboxes, highest_confidence_for_predictions, gt_bbox, pos_idx, high_confidence_indeces, indeces_kept_by_nms, image_info[1])

    def __unnorm_scale_image(self, image):
        """
        Args: image
        return: unnormalized numpy array as uint8
        """
        image = self.unnorm(image)
        image = (image * 255).cpu().numpy().astype(np.uint8)
        return image

    def __get_sorted_predictions(self, bbox_predictions, classification_predictions, image_info):
        """
        Returns the predicted bboxes, class ids and confidences sorted by confidence and above
        a given threshold
        """
        prediction_bboxes, prediction_confidences = self.__convert_output_to_workable_data(
            bbox_predictions, classification_predictions, image_info[1])

        prediction_bboxes, prediction_confidences = self.__predictions_over_threshold(
            prediction_bboxes, prediction_confidences)

        prediction_bboxes, predicted_classes, highest_confidence_for_predictions, high_confidence_indeces = self.__sort_predictions_by_confidence(
            prediction_bboxes, prediction_confidences)

        highest_confidence_for_predictions = np.reshape(
            highest_confidence_for_predictions, (highest_confidence_for_predictions.shape[0], 1))

        return prediction_bboxes, predicted_classes, highest_confidence_for_predictions, high_confidence_indeces

    def __predictions_over_threshold(self, prediction_bboxes, predicted_confidences):
        """
        keep predictions above a confidence threshold
        """
        keep_indices = []
        for index, one_hot_pred in enumerate(predicted_confidences):
            max_conf = np.amax(one_hot_pred)
            if max_conf > self.confidence_threshold:
                keep_indices.append(index)

        prediction_bboxes = prediction_bboxes[keep_indices]
        predicted_confidences = predicted_confidences[keep_indices]

        return prediction_bboxes, predicted_confidences

    def __sort_predictions_by_confidence(self, prediction_bboxes, prediction_confidences):
        predicted_classes, highest_confidence_for_predictions = self.__get_predicted_class(
            prediction_confidences)
        permutation = (-highest_confidence_for_predictions).argsort()

        highest_confidence_for_predictions = highest_confidence_for_predictions[permutation]
        prediction_bboxes = prediction_bboxes[permutation]
        predicted_classes = predicted_classes[permutation]

        return prediction_bboxes, predicted_classes, highest_confidence_for_predictions, permutation

    def __get_predicted_class(self, prediction_confidences):
        """
        returns class id and value of maximum confidence
        """
        predicted_idxs = np.argmax(prediction_confidences, axis=1)
        idx2id, predicted_classes = {0: 1, 1: 3}, []
        for x in predicted_idxs:
            predicted_classes.append(idx2id[x])
        predicted_classes = np.array(predicted_classes)
        predicted_classes = np.reshape(predicted_classes, (predicted_classes.shape[0], 1))

        highest_confidence_for_predictions = np.amax(
            prediction_confidences, axis=1)

        return predicted_classes, highest_confidence_for_predictions

    def __convert_bboxes_to_workable_data(self, prediction_bboxes, size):
        height, width = size
        prediction_bboxes = activations_to_bboxes(
            prediction_bboxes, self.anchors_hw, self.grid_sizes)
        return self.__rescale_bboxes(prediction_bboxes, size)

    def __convert_confidences_to_workable_data(self, prediction_confidences):
        return prediction_confidences.sigmoid().cpu().numpy()

    def __convert_output_to_workable_data(self, model_output_bboxes, model_output_confidences, size):
        prediction_bboxes = self.__convert_bboxes_to_workable_data(
            model_output_bboxes, size)
        prediction_confidences = self.__convert_confidences_to_workable_data(
            model_output_confidences)
        return prediction_bboxes, prediction_confidences

    def __rescale_bboxes(self, bboxes, size):
        """
        Args: array of bboxes in corner format
        returns: bboxes upscaled by height and width as numpy array on cpu
        """
        height, width = size
        bboxes[:, 0] *= height
        bboxes[:, 2] *= height
        bboxes[:, 1] *= width
        bboxes[:, 3] *= width
        return (bboxes.cpu().numpy()).astype(int)


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor
