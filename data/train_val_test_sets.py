# template to construct the dataloaders 


def split_image_data(train_dataset,
                     valid_dataset=None,
                     test_dataset=None,
                     batch_size,
                     num_workers,
                     valid_size=0.1,
                     sampler=SubsetRandomSampler):
    
    n_train = len(train_dataset)
    indices = list(range(n_train))
    
    train_sampler = sampler(indices)
    
    np.randomshuffle(indices)
    
    split = int(np.floor(valid_size*n_train))
    
    if valid_dataset is not None:
        valid_loader = DataLoader(valid_dataset, batch_size=1, num_workers=num_workers)
    else:
        train_idx, val_idx = indices[split:], indices[:split]
        train_sampler = sampler(train_idx)
        val_sampler = sampler(val_idx)
        valid_loader = DataLoader(train_dataset,batch_size=1,num_workers=num_workers)
        
        
    train_loader = DataLoader(train_dataset,batch_size=batch_size,sanpler=train_sampler,num_workers=num_workers)
    return train_loader,valid_loader,test_loader