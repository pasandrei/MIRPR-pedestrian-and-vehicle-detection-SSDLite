from general_config import general_config, constants

model_id = general_config.model_id

# 100 - background id
complete_training_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90, 100]
only_people = [1, 100]

model_to_ids = {
    constants.ssd_modified: only_people,
    constants.ssd: complete_training_ids,
    constants.ssdlite: complete_training_ids
}

training_ids = model_to_ids[model_id]

# these can be customized if evaluation only on a subset of classes is desired
eval_cat_ids = training_ids[:-1]

training_ids2_idx = {training_ids[i]: i for i in range(len(training_ids))}

idx_training_ids2 = {i: training_ids[i] for i in range(len(training_ids))}

person_ids = set([0])
vehicle_ids = set([1, 2, 3, 4, 5, 6, 7, 8])

person_color = {k: (0, 255, 0) for k in person_ids}
vehicle_color = {k: (0, 0, 255) for k in vehicle_ids}

person_color.update(vehicle_color)
complete_map = person_color
