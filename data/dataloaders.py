def get_datasets():
    # create data loaders
    train_set = dataset.CityscapeDataset(path=Path, dataset_type='train')
    
    # val n test sets..
    
    
    return split_image_data(train_set,val_set,test_set)