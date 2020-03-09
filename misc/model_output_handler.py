import torch
import numpy as np
import copy

from general_config.anchor_config import default_boxes
from utils.box_computations import wh2corners_numpy, corners_to_wh
from utils.postprocessing import nms


class Model_output_handler():
    """
    Class used to bring the raw model outputs to interpretable data
    -> bbox coordinates and respective predicted class
    """
    def __init__(self, params):
        self.params = params
        self.unnorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        self.confidence_threshold = params.conf_threshold
        self.suppress_threshold = params.suppress_threshold

        self.anchors_xywh = default_boxes(order="xywh")
        self.anchors_xywh = self.anchors_xywh.to('cpu')

        self.scale_xy = 10
        self.scale_wh = 5

    def process_outputs(self, bbox_predictions, classification_predictions, image_info):
        """
        returns complete model outputs in format array of:
        bbox, class id, confidence
        all operations done on cpu
        """
        prediction_bboxes, predicted_classes, highest_confidence_for_predictions, _ = self._get_sorted_predictions(
            bbox_predictions, classification_predictions, image_info)

        # convert to corners for nms
        prediction_bboxes = wh2corners_numpy(prediction_bboxes[:, :2], prediction_bboxes[:, 2:])
        indeces_kept_by_nms = nms(prediction_bboxes, predicted_classes, self.suppress_threshold)

        # new structure: array of bbox, class, confidence
        prediction_bboxes = corners_to_wh(prediction_bboxes)
        complete_outputs = np.concatenate(
            (prediction_bboxes, predicted_classes, highest_confidence_for_predictions), axis=1)

        return complete_outputs[indeces_kept_by_nms]

    def _unnorm_scale_image(self, image):
        """
        Args: image
        return: unnormalized numpy array as uint8
        """
        image = self.unnorm(image)
        image = (image * 255).cpu().numpy().astype(np.uint8)
        return image

    def _get_sorted_predictions(self, bbox_predictions, classification_predictions, image_info):
        """
        Returns the predicted bboxes, class ids and confidences sorted by confidence and above
        a given threshold
        """
        prediction_bboxes, prediction_confidences = self._convert_output_to_workable_data(
            bbox_predictions, classification_predictions, image_info[1])

        prediction_bboxes, prediction_confidences = self._predictions_over_threshold(
            prediction_bboxes, prediction_confidences)

        prediction_bboxes, predicted_classes, highest_confidence_for_predictions, high_confidence_indeces = self._sort_predictions_by_confidence(
            prediction_bboxes, prediction_confidences)

        highest_confidence_for_predictions = np.reshape(
            highest_confidence_for_predictions, (highest_confidence_for_predictions.shape[0], 1))

        return prediction_bboxes, predicted_classes, highest_confidence_for_predictions, high_confidence_indeces

    def _predictions_over_threshold(self, prediction_bboxes, predicted_confidences):
        """
        keep predictions above a confidence threshold
        """
        highest_confidence = np.amax(predicted_confidences, axis=1)
        keep_indices = (highest_confidence > self.confidence_threshold)

        prediction_bboxes = prediction_bboxes[keep_indices]
        predicted_confidences = predicted_confidences[keep_indices]

        return prediction_bboxes, predicted_confidences

    def _sort_predictions_by_confidence(self, prediction_bboxes, prediction_confidences):
        predicted_classes, highest_confidence_for_predictions = self._get_predicted_class(
            prediction_confidences)
        permutation = (-highest_confidence_for_predictions).argsort()

        highest_confidence_for_predictions = highest_confidence_for_predictions[permutation]
        prediction_bboxes = prediction_bboxes[permutation]
        predicted_classes = predicted_classes[permutation]

        return prediction_bboxes, predicted_classes, highest_confidence_for_predictions, permutation

    def _get_predicted_class(self, prediction_confidences):
        """
        returns class idx and value of maximum confidence
        """
        predicted_idxs = np.argmax(prediction_confidences, axis=1)
        predicted_idxs = np.reshape(predicted_idxs, (predicted_idxs.shape[0], 1))

        highest_confidence_for_predictions = np.amax(prediction_confidences, axis=1)

        return predicted_idxs, highest_confidence_for_predictions

    def _convert_offsets_to_bboxes(self, prediction_bboxes, size):
        """
        Computes offsets according to the ssd paper formula
        """
        prediction_bboxes = prediction_bboxes.cpu()

        prediction_bboxes[:, :2] = (1/self.scale_xy)*prediction_bboxes[:, :2]
        prediction_bboxes[:, 2:] = (1/self.scale_wh)*prediction_bboxes[:, 2:]

        prediction_bboxes[:, :2] = prediction_bboxes[:, :2] * self.anchors_xywh[:, 2:] + \
            self.anchors_xywh[:, :2]
        prediction_bboxes[:, 2:] = prediction_bboxes[:, 2:].exp() * self.anchors_xywh[:, 2:]

        return self._rescale_bboxes(prediction_bboxes, size)

    def _convert_confidences_to_workable_data(self, prediction_confidences):
        """
        Applies softmax or sigmoid respectively to confidence predictions
        """
        if self.params.loss_type == "BCE":
            return prediction_confidences.sigmoid().cpu().numpy()
        else:
            prediction_confidences = torch.nn.functional.softmax(prediction_confidences, dim=1)
            # want actual object probabilities, so cut the background column
            return prediction_confidences[:, :-1].cpu().numpy()

    def _convert_output_to_workable_data(self, model_output_bboxes, model_output_confidences, size):
        prediction_bboxes = self._convert_offsets_to_bboxes(
            model_output_bboxes, size)
        prediction_confidences = self._convert_confidences_to_workable_data(
            model_output_confidences)
        return prediction_bboxes, prediction_confidences

    def _rescale_bboxes(self, bboxes, size):
        """
        Arguments:
        bboxes - bboxes to be upscaled
        size - original size of the image used for upscaling
        """
        width, height = size
        scale_bboxes = copy.deepcopy(bboxes)
        scale_bboxes[:, 0] *= width
        scale_bboxes[:, 2] *= width
        scale_bboxes[:, 1] *= height
        scale_bboxes[:, 3] *= height
        return (scale_bboxes.cpu().numpy()).astype(int)


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
