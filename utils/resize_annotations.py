import json
from pathlib import Path


def resize_from_annotations_file(annotations_source, annotations_destination,
                                 target_width=300, target_height=300):
    """
    Args:
    - annotations_source: path to the annotations file to be modified
    - annotations_destination: path to where the resized annotations file should go
    - (target_width, target_height): integers that represent images' height and with after resize

    Explanation:
    Saves resized images and corresponding annotations in a new file

    Returns: None
    """

    print(annotations_source)
    with open(annotations_source, 'r') as annotations_file:
        data = json.load(annotations_file)

        image_id_to_image_info = {}
        for image in data['images']:
            image_id_to_image_info[image['id']] = image

        for annotation in data['annotations']:
            image_id = annotation['image_id']
            image_info = image_id_to_image_info[image_id]

            annotation['bbox'][0] = annotation['bbox'][0] / image_info['width'] * target_width
            annotation['bbox'][2] = annotation['bbox'][2] / image_info['width'] * target_width
            annotation['bbox'][1] = annotation['bbox'][1] / image_info['height'] * target_height
            annotation['bbox'][3] = annotation['bbox'][3] / image_info['height'] * target_height

        for image in data['images']:
            image['width'] = target_width
            image['height'] = target_height

        with open(annotations_destination, 'w') as outfile:
            json.dump(data, outfile)


path = Path('C:\\Users\\Andrei Popovici\\Desktop\\COCO')

train_annotations_path = path / 'annotations' / 'instances_train2017.json'
train_annotations_path_destionation = path / 'annotations' / 'instances_train2017_resized.json'
resize_from_annotations_file(train_annotations_path, train_annotations_path_destionation)

val_annotations_path = path / 'annotations' / 'instances_val2017.json'
val_annotations_path_destionation = path / 'annotations' / 'instances_val2017_resized.json'
resize_from_annotations_file(val_annotations_path, val_annotations_path_destionation)
