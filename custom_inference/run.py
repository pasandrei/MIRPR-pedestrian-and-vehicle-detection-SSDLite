import torch
import cv2
import torchvision.transforms.functional as F
from pathlib import Path

from train.params import Params
from misc.model_output_handler import Model_output_handler
from general_config import constants, general_config
from utils import training
from utils.postprocessing import nms, postprocess_until_nms
from utils.box_computations import wh2corners_numpy


class Custom_Infernce():
    def __init__(self, model_id="ssdlite_1_class"):
        self.model_id = model_id
        self.params = Params(constants.params_path.format(model_id))

        self.model = training.model_setup(self.params)
        self.model = training.load_weigths_only(self.model, self.params)
        self.model = self.model.to(general_config.device)
        self.model.eval()

        self.output_handler = Model_output_handler(self.params)
        self.source_dir = Path.cwd() / "custom_inference" / "samples"
        self.save_dir = Path.cwd() / "custom_inference" / "outputs"

        self.source_video_dir = Path.cwd() / "custom_inference" / "video_sample" / "video.mp4"
        self.save_video_dir = Path.cwd() / "custom_inference" / "video_output" / "video.mp4"

    def run_inference(self, image, modify_image=True, custom_settings=None):
        """
        If modify_image flag is True, the boxes are drawn on the given image, otherwise this
        function just returns the predicted boxes

        custom_settings, if set, should be a tuple of (nms_threshold, conf_threshold), this
        !!! overwrites the original settings
        """
        if custom_settings:
            print("Current custom settings: ", custom_settings)
            nms_thresh, conf_thresh = custom_settings
            self.output_handler.suppress_threshold = nms_thresh
            self.output_handler.conf_threshold = conf_thresh
        with torch.no_grad():
            original_image = image.copy()
            heigth, width, _ = original_image.shape

            image = cv2.resize(image, (300, 300))
            image = F.to_tensor(image)
            image = F.normalize(image, mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
            image = image.to(general_config.device)
            image = image.unsqueeze(dim=0)
            boxes, confs = self.model(image)
            boxes = boxes.squeeze().permute(1, 0)
            confs = confs.squeeze().permute(1, 0)

            boxes, classes = postprocess_until_nms(self.output_handler, boxes,
                                                   confs, (width, heigth))

            boxes = wh2corners_numpy(boxes[:, :2], boxes[:, 2:])
            kept_indeces = nms(boxes, classes, self.output_handler.suppress_threshold)

            boxes = boxes[kept_indeces].astype(int)
            if modify_image:
                image = self.plot_boxes(original_image, boxes)
                return image
            return boxes

    def run_image(self):
        print("Source directory: ", self.source_dir)
        for img_path in self.source_dir.glob('*'):
            image = cv2.imread(str(img_path))
            image = self.run_inference(image)
            save_path = self.save_dir / (img_path.stem + img_path.suffix)
            cv2.imwrite(str(save_path), image)

    def run_video(self):
        print("Source directory: ", self.source_video_dir)
        vidcap = cv2.VideoCapture(str(self.source_video_dir))
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        print("Video FPS: ", fps)
        success, image = vidcap.read()
        heigth, width, _ = image.shape
        size = (width, heigth)
        out = cv2.VideoWriter(str(self.save_video_dir), cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
        while success:
            success, image = vidcap.read()
            if not success:
                break
            image = self.run_inference(image)
            out.write(image)
        out.release()

    def plot_boxes(self, image, boxes):
        color = (0, 255, 0)
        for x1, y1, x2, y2 in boxes:
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        return image
