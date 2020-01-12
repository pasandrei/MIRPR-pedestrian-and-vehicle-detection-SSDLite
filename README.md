# Pedestrian and vehicle detection with SSDLite

We achieved 53% AP50 for Medium and Large objects on COCO dataset (Average Precision with IoU 50%).

We trained the network only on humans and car annotations that are considered medium (32\*32 < area < 96\*96) or large (96\*96 < area). We used modified anchors on grid sizes 1x1, 2x2, 3x3, 5x5, 10x10.

### Future work

In the following iterations we will use data augmentation because the network overfits the training data.

We are also planning to test the netowrk performance with 20x20 anchors.
