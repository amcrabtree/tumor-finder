
import os
import sys
import numpy as np
import pandas as pd
#from tumor_utils.tiling import tile_unlabeled_wsi
import json
import torch
from torch.utils.data import DataLoader
from tumor_utils import data, tformer # custom class for loading & transforming data
from tumor_utils.data import UnlabeledImgData # custom class for unlabeled images
from tumor_utils.test import Tester # custom class for testing
from tumor_utils.wsi import WSI # custom WSI class
import glob

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

wsi_dir="/projects/bgmp/acrabtre/tumor-finder/data/tcga"
meta_file="/projects/bgmp/acrabtre/tumor-finder/data/tcga/tcga_meta.txt"
tile_dir="/projects/bgmp/acrabtre/tumor-finder/data/tcga/tiles"
model_file="/projects/bgmp/acrabtre/tumor-finder/output/WSI_VGG_Adam/running_model.pt"
tiling=True



def tile_wsi_list(wsi_file_list, tile_dir):
    for wsi_file in wsi_file_list:
        WSI(wsi_file)
        print(f"\nTiling file: {os.path.basename(wsi_file)}")
        WSI.generate_tiles(outdir=tile_dir,tile_size=256,level=0)


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
        #self.config = self.parse_config()
        self.tile_files = self.list_tile_files()
        self.dataset = self.load_data()
        self.pred_np = self.make_preds()
        self.label_dict = {0:'normal', 1:'tumor'}
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
        pattern=os.path.join(self.tile_dir, f"**/*{wsi_name}*.*")
        file_list = glob.glob(pattern)
        return file_list

    def load_data(self):
        """ Returns pytorch DataLoader object. """
        image_transform = tformer.custom_tfm(
            data='image', model= "vgg16", input_size=256)
        # define and transform images 
        dataset = UnlabeledImgData(
            self.tile_files,
            transform = image_transform)
        return dataset

    def make_preds(self):
        """ Makes predictions of whether tiles are tumor or normal. 
        """
        # Load trained model from file
        model = torch.load(self.model_file)
        model.to(device)
        model.eval() # puts model into inference mode
        
        torch.set_grad_enabled(False) 
        pred_list=[]
        for image in enumerate(self.dataset):
            image = image.to(device)
            output = self.model(image)
            pred = output.argmax(-1).cpu().numpy()
            prob = torch.nn.functional.softmax(output, dim=1)[:, 1].cpu()
            print(f'pred: {pred} prob: {prob}')
            pred_list.append(pred)
        pred_np = np.asarray(pred_list)
        return pred_np
    
    def interpret_preds(self)->list:
        """ Convert prediction categories to labels.
        """
        label_dict = self.label_dict
        idx = np.searchsorted(self.label_dict.keys(),self.pred_np)
        label_list = np.asarray(label_dict.values())[idx].tolist()
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



if __name__=='__main__':

    # import metadata as pandas df and save file list
    meta_df = pd.read_csv(meta_file, sep='\t')
    meta_df['pathname'] = os.path.join(wsi_dir, meta_df['slide_id']+".svs")
    wsi_file_list = [x for x in meta_df['pathname']]
    wsi_stage_list = [x for x in meta_df['stage']]
    print("meta_df:\n", meta_df)

    # tile wsis if needed
    if tiling==True: tile_wsi_list(wsi_file_list, wsi_dir, tile_dir)

    # run model on all its tiles and save numbers of normal vs. tumor tiles
    preds_dict = {} # make dict: key=wsi file value=(n_tumor_tiles,n_normal_tiles)
    for i, wsi_file in enumerate(wsi_file_list): 
        model_preds = TumorPredWSI(wsi_file, tile_dir, model_file) 
        n_tumor = model_preds.pred_count_dict['tumor']
        n_normal = model_preds.pred_count_dict['normal']
        t_purity = round(n_tumor/(n_tumor+n_normal),4)
        preds_dict[i] = [wsi_file, n_tumor, n_normal, t_purity]
    preds_df = pd.DataFrame.from_dict(
        preds_dict, 
        orient="index",
        columns=['wsi_file','n_tumor','n_normal','t_purity'])
    print("preds_df:\n", preds_df)

    # perform rank correlation using WSI stage vs. WSI tumor purity
    #rank_df = pd.concat([meta_df, preds_df], ignore_index=True, sort=False)
    rank_df = pd.DataFrame().assign(
        t_purity = preds_df['t_purity'], 
        stage = meta_df['stage'])
    print("rank_df:\n", rank_df)
    print(f'\n\tTotal rank correlation value is: {rank_df.corr()}')
