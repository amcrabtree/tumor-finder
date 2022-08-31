
import os
import sys
import numpy as np
import pandas as pd
from PIL import Image
import json
import torch
from tumor_utils import tformer # custom class for loading & transforming data
from tumor_utils.data import UnlabeledImgData # custom class for unlabeled images
import glob
from torchvision import transforms
from torch.utils.data import DataLoader

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class TumorPredWSI():
    """ 
    Produces a tile prediction object which uses an ML model to produce tile predictions
    from a single WSI. 

    Data attributes:
        id_coord: top left tile coordinate (in image coordinate system)
        size: desired output tile size in pixels (h=w)
        level: desired output level (0 is highest quality level)
        points: list of the 4 xy coordinates comprising tile corners at level 0
        label: labels tile according to ROIs ('tumor', 'normal' or 'margin')
        np_img: numpy array of tile image 
       
    Methods:
        ReturnPoints(): Return points attribute, adjusted to level 0 coordinates
        TileLabel(): Returns tile label, according to annotated ROIs.
        TileImg(): Converts tile to numpy array.

    Usage:
    >>> new_tile = Tile(ann_wsi, id_coord=[4600,15000], size=256, level=4)
    """
    def __init__(self, wsi_file:str, tile_dir:str, model_file:str):
        self.wsi_file = wsi_file
        self.tile_dir = tile_dir
        self.model_file = model_file
        self.label_dict = {0:'normal', 1:'tumor'}
        #self.config = self.parse_config()
        self.tile_files = self.list_tile_files()
        self.dataloader = self.load_data()
        self.pred_np = self.make_preds()
        self.pred_labels = self.interpret_preds()
        self.pred_count_dict = self.count_preds()

    def parse_config(self):
        # initial run configurations 
        config_file = open(sys.argv[1], "r")
        config = json.loads(config_file.read())
        config_file.close()

        # set full output filenames
        project_dir = os.path.join(config['run_dir'], config['run_name'])
        for el in config['output']:
            config['output'][el] = os.path.join(project_dir, config['output'][el])

    def list_tile_files(self)->list:
        """ Returns list of tile filenames in Dataset
        """
        wsi_name=os.path.basename(self.wsi_file).split(".")[0]
        pattern=os.path.join(self.tile_dir, f"*{wsi_name}*.*")
        file_list = glob.glob(pattern)
        return file_list

    def load_data(self):
        """ Returns pytorch DataLoader object. """
        image_transform = tformer.custom_tfm(data='image', model= "vgg16", input_size=256)
        # define and transform images 
        dataset = UnlabeledImgData(
            self.tile_files,
            transform = image_transform)
        #print("torch size:", dataset[0].size())
        dataloader = DataLoader(
            dataset, 
            batch_size = 128, 
            num_workers = 1, 
            pin_memory = True,
            shuffle = False, 
            drop_last = True)

        return dataloader

    def make_preds(self):
        """ Makes predictions of whether tiles are tumor or normal. 
        """
        # Load trained model from file
        model = torch.load(self.model_file)
        model.to(device)
        model.eval() # puts model into inference mode
        
        torch.set_grad_enabled(False) 
        img_path = "./data/tiles/test/normal/wsi_patient_073_node_1_tile_29710_label_normal.png"
        image = Image.open(img_path)
        image = np.array(image)
        #transform = tformer.custom_tfm(data='image', model= "vgg16", input_size=256)
        transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.CenterCrop(224),
                    transforms.ConvertImageDtype(torch.float), 
                    transforms.Normalize(
                        mean=[0.48235, 0.45882, 0.40784], 
                        std=[0.00392156862745098, 0.00392156862745098, 0.00392156862745098]),
                ])
        image = transform(image)
        image = image.to(device)
        outputs = model(image)
        pred = outputs.argmax(-1).cpu()
        print("pred:", pred)
        return pred
    
    def interpret_preds(self)->list:
        """ Convert prediction categories to labels.
        """
        label_list = np.vectorize(self.label_dict.get)(self.pred_np).tolist()
        return label_list
    
    def count_preds(self)->tuple:
        """ Save binary predictions as tuple. 
        """
        pred_count_dict = {}
        for pred in self.pred_labels:
            if pred not in pred_count_dict:
                pred_count_dict[pred]=0
            else:
                pred_count_dict[pred]+=1
        return pred_count_dict

    def make_pred_df(self, outfile:str=""):
        """ Save relevant prediction data to dataframe and save if desired. 
        """
        df = pd.DataFrame ({
            'model':torch.load(self.model_file).__class__.__name__,
            'wsi_file':self.wsi_file,
            'tile_file':self.tile_files,
            'pred':self.pred_np,
            'label':self.pred_labels})
        if outfile!="": df.to_csv(outfile, index=False)