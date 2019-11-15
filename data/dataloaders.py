import torchvision
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from data.dataset import CocoDetection


def get_dataloaders():
    ''' creates and returns train and validation data loaders '''

    composed_transforms = transforms.Compose([transforms.ToTensor()])

    # train_dataset = torchvision.datasets.CocoDetection(root='../../COCO/train2017/train2017',
    #                                      annFile='../../COCO/annotations_trainval2017/annotations/instances_train2017.json',
    #                                       transform=composed_transforms)

    # train_dataloader = DataLoader(train_dataset, batch_size=1,
    #                        shuffle=True, num_workers=0)

    validation_dataset = CocoDetection(root='C:\\Users\\Andrei Popovici\\Desktop\\COCO_new\\val2017',
                                       annFile='C:\\Users\\Andrei Popovici\\Desktop\\COCO_new\\annotations\\instances_val2017.json',
                                       transform=composed_transforms, transforms=None)

    valid_dataloader = DataLoader(validation_dataset, batch_size=1,
                                  shuffle=False, num_workers=0)

    return valid_dataloader, valid_dataloader
