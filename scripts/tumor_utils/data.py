import os
import glob
import re
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image

class TiledDataset(Dataset):
    """ Generates PyTorch Dataset object for model training

    :param subset_dir: Directory with tiles in labeled subfolders
    :type subset_dir: str
    :param transform: Optional transform to be applied to the dataset, defaults to None. 
    :type transform: torchvision.transforms, optional

    """
    def __init__(self, set_dir, transform=None, target_transform=None):
        """ Constructor method
        """
        self.subset_dir = set_dir
        self.transform = transform
        self.target_transform = target_transform
        self.label_dict = {0:'normal', 1:'tumor'}
        self.tile_files = self.TileFileList()
        self.all_labels = self.LabelsList()
        
    def TileFileList(self)->list:
        """ Returns list of tile filenames in Dataset
        """
        pattern=os.path.join(self.subset_dir, "**/*.*")
        file_list = glob.glob(pattern)
        return file_list

    def LabelsList(self)->list:
        """ Returns list of tile labels in Dataset
        """
        pattern=re.compile('.+_label_(\w+)[.]\w+')
        label_list = [pattern.search(f).group(1) for f in self.tile_files]
        return label_list

    def __len__(self):
        """ Returns number of observations (images/labels) 
        """
        return len(self.all_labels)

    def __getitem__(self, idx):
        """ Returns a tuple of image & label for a given index 

        :param idx: index of the image/label pair
        :type idx: pytorch tensor

        :return: (image, label)
        :rtype: tuple

        """
        if torch.is_tensor(idx): idx = idx.tolist()
            
        # define where images will be found
        img_path = self.tile_files[idx]
        image = Image.open(img_path)

        # define what the labels are and convert to numeric 
        label = self.all_labels[idx]
        label_num = list(self.label_dict.keys()) [ 
            list(self.label_dict.values()).index(label)]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label_num = self.target_transform(label_num)
        #print(f"\n\ttype of image: {image.size()}")
        #print(f"\ttype of label_num: {type(label_num)}")

        return (image, label_num)



def TryLoader(loader:DataLoader):
    """ Ensure the dataset and dataloader is working properly. 
    Note: wasn't working after I transformed dataset. Didn't care to fix. 
    """
    import matplotlib.pyplot as plt
    train_img, train_labels = next(iter(loader))
    transform = transforms.ToPILImage()
    img = transform(train_img)
    print(f"Label: {label}")
    print(f"Feature batch shape: {train_img.size()}")
    print(f"Labels batch shape: {train_labels.size()}")
    img = train_img[0].permute(1,2,0)
    label = train_labels[0]
    print(f"Label: {label}")
    plt.imshow(img)
    plt.show()
