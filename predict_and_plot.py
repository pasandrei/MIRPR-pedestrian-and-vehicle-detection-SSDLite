from misc.postprocessingt import nms


def model_output_pipeline(params_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    params = Params(params_path)

    if params.model_id == 'ssdnet':
        model = SSDNet.SSD_Head()
    model.to(device)

    checkpoint = torch.load('misc/experiments/{}/model_checkpoint'.format(params.model_id))
    model.load_state_dict(checkpoint['model_state_dict'])
    print('Model loaded successfully')

    _, valid_loader = dataloaders.get_dataloaders(params)

    for batch_images, batch_targets in valid_loader:
        batch_images.to(device)

        # predictions[0] = B x #anchors x 4
        # predictions[1] = B x #anchors x 3 -> [0.2, 0.1, 0.9], [0.01, 0.01, 0.8]
        predictions = model(batch_images)

        # move everything to cpu for plotting
        batch_images = batch_images.cpu()
        predictions[0] = predictions[0].cpu()
        predictions[1] = predictions[1].cpu()

        for idx in range(len(batch_images)):
            current_image = batch_images[idx]

            current_image_bboxes = batch_targets[0][idx]
            current_image_class_ids = batch_targets[1][idx]

            current_prediction_bboxes = predictions[0][idx]
            current_prediction_class_ids = predictions[1][idx]

            plot_model_outputs(current_image, current_image_bboxes, current_image_class_ids,
                               current_prediction_bboxes, current_prediction_class_ids)


def plot_model_outputs(current_image, current_image_bboxes, current_image_class_ids,
                       current_prediction_bboxes, current_prediction_class_ids):
    """

    """
    keep_indices = []
    for idx, one_hot_pred in enumerate(current_prediction_class_ids):
        max_confidence, position = current_prediction_class_ids.max(dim=0)
        if position != 2:
            keep_indices.append(position)

    current_prediction_bboxes = current_prediction_bboxes.numpy()
    keep_indices = numpy.array(keep_indices)

    kept_bboxes = current_prediction_bboxes[keep_indices]
    # RIPPP
    post_nms_bboxes = nms(kept_bboxes)
