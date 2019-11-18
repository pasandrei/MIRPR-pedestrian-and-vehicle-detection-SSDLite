import json
import math
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def extract_from_annotations_file(annotations_file_path, folder, wanted_categories_id):
    print(annotations_file_path)
    print(folder)

    ratios = []

    with open(annotations_file_path, 'r') as annotations_file:
        data = json.load(annotations_file)

        for annotation in data['annotations']:
            if annotation['category_id'] in wanted_categories_id:
                width = annotation['bbox'][2]
                height = annotation['bbox'][3]

                ratio = height/width
                ln_ratio = math.log(ratio)

                ratios.append(ln_ratio)

    ratios = np.array(ratios)
    plt.hist(ratios, bins=20)
    plt.axvline(ratios.mean(), color='k', linestyle='dashed', linewidth=1)
    print(math.exp(ratios.mean()))
    plt.show()


# path = sys.argv[1]
path = Path('C:\\Users\\Andrei Popovici\\Desktop\\COCO_new')

wanted_categories_id = [3]


val_annotations_path = path / 'annotations' / 'instances_val2017.json'
val_folder_path = path / 'val2017'
extract_from_annotations_file(val_annotations_path, val_folder_path, wanted_categories_id)
