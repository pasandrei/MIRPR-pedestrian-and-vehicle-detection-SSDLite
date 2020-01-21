from pathlib import Path

dataset_root = Path('C:\\Users\Andrei Popovici\Desktop\COCO')

train_annotations_path = dataset_root / 'annotations/instances_train2017.json'
train_images_folder = dataset_root / 'train2017'

val_annotations_path = dataset_root / 'annotations/instances_val2017.json'
val_images_folder = dataset_root / 'val2017'
