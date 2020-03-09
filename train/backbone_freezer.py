

class Backbone_Freezer():
    """
    Handles the freezing/unfreezing of the backbone of the model dynamically during training
    works for MobileNetV2
    """
    def __init__(self, params, freeze_idx=19):
        self.params = params
        self.freeze_idx = freeze_idx

    def freeze_backbone(self, model):
        for params in model.backbone.parameters():
            params.requires_grad = False

    def unfreeze_from(self, layer_idx, model):
        """
        MobileNetV2 has 19 residual bottleneck layers
        unfreezes layers from layer_idx to 18
        """
        for i in range(layer_idx, 19):
            for parameters in model.backbone.features[i].parameters():
                parameters.requires_grad = True

    def step(self, epoch, model):
        """
        uniformly unfreeze 15 layers from the start to the first decay
        unfreeze the rest of the layers as well at the second_decay
        if backbone was not frozen, this has no effect
        """
        if epoch <= self.params.first_decay:
            freeze_idx = int(19 - (15 / self.params.first_decay) * epoch)
            self.unfreeze_from(freeze_idx, model)

        # when the second decay is reached, unfreeze the final layers as well
        if epoch == self.params.second_decay:
            self.unfreeze_from(0, model)
