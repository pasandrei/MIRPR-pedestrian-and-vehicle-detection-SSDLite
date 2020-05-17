from pathlib import Path

dataset_root = Path('C:/Users/Dorel/Documents/COCO')

train_annotations_path = dataset_root / 'annotations/instances_train2017.json'
train_images_folder = dataset_root / 'train2017'

val_annotations_path = dataset_root / 'annotations/instances_val2017.json'
val_images_folder = dataset_root / 'val2017'

test_annotations_path = dataset_root / 'annotations/image_info_test-dev2017.json'
test_images_folder = dataset_root / 'test2017'

params_path = 'misc/experiments/{}/params.json'
stats_path = 'misc/experiments/{}/stats.json'
model_path = 'misc/experiments/{}/model_checkpoint'
model_path_loss = 'misc/experiments/{}/model_checkpoint_loss'

poly_lr = "poly"
retina_lr = "retina"

BCE_loss = "BCE"
softmax_loss = "softmax"

ssdlite = "ssdlite"
ssd = "resnetssd"
ssd_modified = "ssdlite_1_class"
