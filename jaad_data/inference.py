import torchvision.transforms.functional as F
from PIL import Image
import os
import cv2
import copy
import numpy as np
import torch

from misc import postprocessing
from misc.utils import *

rootdir = '.'


def jaad_inference(model, output_handler):

    for subdir, _, files in os.walk(rootdir):
        new_dir_name = subdir + "_inference"
        try:
            subdir.index("video")
            v = 0
            try:
                subdir.index("inference")
            except Exception as ex:
                v = 1
            if not v:
                raise Exception()
            os.mkdir(new_dir_name)
        except Exception as e:
            print(new_dir_name + " already exists or does not contain images")
            continue
        for file in files:
            image_path = os.path.join(subdir, file)
            img = cv2.imread(image_path)  # image as numpy array
            img = feed_to_model(model, img, output_handler)
            inference_image_path = os.path.join(new_dir_name, file)
            cv2.imwrite(inference_image_path, img)


def feed_to_model(model, img, output_handler):
    with torch.no_grad():
        model.eval()
        init_size = img.shape
        img = Image.fromarray(img.astype('uint8'), 'RGB')
        img = F.resize(img, size=(320, 320), interpolation=2)
        img = F.to_tensor(img)
        img = F.normalize(img, mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
        # add batch channel
        img = img.view((1, img.shape[0], img.shape[1], img.shape[2]))

        predictions = model(img)

        prediction_bboxes, predicted_classes, _, _ = output_handler._get_sorted_predictions(
            predictions[0][0], predictions[1][0], (0, (init_size[0], init_size[1])))

        indeces_kept_by_nms = postprocessing.nms(prediction_bboxes, predicted_classes,
                                                 output_handler.suppress_threshold)

        final_bbox = prediction_bboxes[indeces_kept_by_nms]
        final_class = predicted_classes[indeces_kept_by_nms]

        img = img.view((img.shape[1], img.shape[2], img.shape[3]))
        img = output_handler._unnorm_scale_image(img)

        return plot_bounding_boxes(image=img, bounding_boxes=final_bbox, classes=final_class,
                                   bbox_type="pred", size=(init_size[0], init_size[1]))
