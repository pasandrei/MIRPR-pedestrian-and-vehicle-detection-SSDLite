class Dataset():
    
    def __init__ (self, path, dataset_type):
        labels_path = '/gtFine'
        images_path = '/leftImg8buit'
            
        # construct paths and read data from them
        self.images = get_files(self.images_path)
        
    def get_files(self, path):
        v = []
        for subdir in os.listdir(path):
            v += [subdir + '/' + f for f in os.listdir(os.path.join(path, subdir))]
        return v
        
    def __len__(self):
        
        # return length of dataset
        return len(self.images)
        
    def __getitem__(self, idx):
        
        # get sample: input_, target for each index
        
        image = io.imread(os.path.join(self.images_path,self.images[idx]))
        
        # apply transform and construct sample dict 
        
        return sample
        
    def augmentation(image, label):
        # use nn.functional 
        
        
        
        
        
        
        
        
        
        
        