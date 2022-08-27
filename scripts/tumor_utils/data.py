import os
import glob
import re
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

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
        self.label_dict = {0:'normal', 1:'tumor'}
        self.tile_files = self.list_tile_files()
        self.all_labels = self.list_labels()
        self.transform = transform
        self.target_transform = target_transform
        
    def list_tile_files(self)->list:
        """ Returns list of tile filenames in Dataset
        """
        pattern=os.path.join(self.subset_dir, "**/*.*")
        file_list = glob.glob(pattern)
        return file_list

    def list_labels(self)->list:
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
        image = np.array(image)

        # define what the labels are and convert to numeric 
        label = self.all_labels[idx]
        label_num = list(self.label_dict.keys()) [ 
            list(self.label_dict.values()).index(label)]

        # transform data if transforms are given
        if self.transform: image = self.transform(image)
        if self.target_transform: label_num = self.target_transform(label_num)

        return (image, label_num)





class PCam(Dataset):
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
        self.transform = None
        self.target_transform = None
        self.label_dict = {0:'normal', 1:'tumor'}
        self.tile_files = self.list_tile_files()
        self.all_labels = self.list_labels()
        self.transform = transform
        self.target_transform = target_transform
    
    def list_tile_files(self)->list:
        """ Returns list of tile filenames in Dataset
        """
        pattern=os.path.join(self.subset_dir, "**/*.*")
        file_list = glob.glob(pattern)
        return file_list

    def list_labels(self)->list:
        """ Returns list of tile labels in Dataset
        """
        label_list = [f.split("/")[-2] for f in self.tile_files]
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
        #if torch.is_tensor(idx): idx = idx.tolist()
            
        # define where images will be found
        img_path = self.tile_files[idx]
        image = Image.open(img_path)
        image = np.array(image)

        # define what the labels are and convert to numeric 
        label = self.all_labels[idx]
        label_num = list(self.label_dict.keys()) [ 
            list(self.label_dict.values()).index(label)]

        # transform data if transforms are given
        if self.transform: image = self.transform(image)
        if self.target_transform: label_num = self.target_transform(label_num)

        return image, label_num

class UnlabeledImgData(Dataset):
    """ Generates PyTorch Dataset object for model training

    :param subset_dir: Directory with tiles in labeled subfolders
    :type subset_dir: str
    :param transform: Optional transform to be applied to the dataset, defaults to None. 
    :type transform: torchvision.transforms, optional

    """
    def __init__(self, tile_file_list, transform=None):
        """ Constructor method
        """
        self.tile_files = tile_file_list
        self.transform = transform

    def __len__(self):
        """ Returns number of observations (images/labels) 
        """
        return len(self.tile_files)

    def __getitem__(self, idx):
        """ Returns a numpy image for a given index, transformed
        if specified. 

        Args:
            idx: index of the image in the file list
        """
        # return image as numpy array
        img_path = self.tile_files[idx]
        image = Image.open(img_path)
        image = np.array(image)
        # transform image if specified
        if self.transform: image = self.transform(image)
        return image
