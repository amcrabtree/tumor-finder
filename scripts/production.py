
import pandas as pd
import torch
from tumor_utils import tformer # custom class for loading & transforming data
from tumor_utils.data import UnlabeledImgData # custom class for unlabeled images
from tumor_utils.tiling import tile_wsi_list # custom tiling function
from tumor_utils.pred import Predictor # custom tumor prediction class
from tumor_utils.viz import save_heatmap # make heatmaps for WSIs
from torch.utils.data import DataLoader
import os
import glob
import numpy as np
import re

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def list_tile_files(wsi_file:str, tile_dir:str)->list:
    """ Returns list of tile filenames in Dataset
    """
    wsi_name=os.path.basename(wsi_file).split(".")[0]
    pattern=os.path.join(tile_dir, f"*{wsi_name}*.*")
    file_list = glob.glob(pattern)
    return file_list

def interpret_preds(pred_np:np.ndarray)->list:
    """ Convert prediction categories to labels.
    """
    label_dict = {0:'normal', 1:'tumor'}
    label_list = np.vectorize(label_dict.get)(pred_np).tolist()
    return label_list
    
def count_preds(pred_labels:list)->tuple:
    """ Save binary predictions as tuple. 
    """
    pred_count_dict = {}
    for pred in pred_labels:
        if pred not in pred_count_dict:
            pred_count_dict[pred]=0
        else:
            pred_count_dict[pred]+=1
    return pred_count_dict

if __name__=='__main__':

    wsi_dir="/projects/bgmp/acrabtre/tumor-finder/data/wsi/tcga-dlbc"
    wsi_ext=".svs" # TCGA files will be ".svs"
    #wsi_dir="/projects/bgmp/acrabtre/tumor-finder/data/wsi/prod_wsi"
    #wsi_ext=".tif" 

    meta_file="/projects/bgmp/acrabtre/tumor-finder/data/wsi/tcga-dlbc/meta_TCGA-DLBC.txt"
    #meta_file="/projects/bgmp/acrabtre/tumor-finder/data/wsi/prod_wsi/meta_prod.txt"
    
    tile_dir="/projects/bgmp/acrabtre/tumor-finder/data/tiles/tcga-dlbc"
    #tile_dir="/projects/bgmp/acrabtre/tumor-finder/data/tiles/prod"

    model_file="/projects/bgmp/acrabtre/tumor-finder/output/WSI_VGG_Adam/running_model.pt"
    tiling=True
    
    # Output files:
    predict_csv="./output/tcga-dlbc/pred_stats_short.csv"
    tumor_coords="./output/tcga-dlbc/tumor_coords.csv"

    # import metadata as pandas df and save file list
    meta_df = pd.read_csv(meta_file, sep='\t')
    meta_df['pathname'] = wsi_dir + "/" + meta_df['filename']
    wsi_file_list = [x for x in meta_df['pathname']]
    wsi_stage_list = [x for x in meta_df['stage']]

    # Tile wsis if needed
    if tiling==True: tile_wsi_list(wsi_file_list, tile_dir, tile_size=256, level=0)

    # Load trained model from file
    model = torch.load(model_file)
    model.to(device)

    # run model on all its tiles and save numbers of normal vs. tumor tiles
    preds_dict = {} # make dict: key=wsi file value=(n_tumor_tiles,n_normal_tiles)
    for i, wsi_file in enumerate(wsi_file_list): 
        tile_file_list = list_tile_files(wsi_file, tile_dir) # Get list of tile files

        # Define and transform datasets 
        image_transform = tformer.custom_tfm(data='image', model= "vgg16", input_size=256)
        dataset = UnlabeledImgData(
                tile_file_list,
                transform = image_transform)

        # Load datasets into dataloaders
        data_loader = DataLoader(
            dataset, 
            batch_size = 128, 
            num_workers = 1, 
            pin_memory = True,
            shuffle = False, 
            drop_last = False)

        # 4. Make predictions (perform forward pass using test images)
        print(f'\nPredicting classes for {wsi_file} ...\n')
        model.eval() # puts model into inference mode
        predictor = Predictor(model)
        wsi_preds = predictor.predict(data_loader)
        pred_label_list = interpret_preds(wsi_preds)
        wsi_pred_dict = count_preds(pred_label_list)
        n_tumor = wsi_pred_dict['tumor']
        n_normal = wsi_pred_dict['normal']
        t_purity = round(n_tumor/(n_tumor+n_normal),4)
        preds_dict[i] = [wsi_file, n_tumor, n_normal, t_purity]

        # save CSV with tumor tile coordinates for heatmap
        pattern = r".+_loc_(\d+)-(\d+)"
        coords = [m for x in tile_file_list for m in re.search(pattern, x).groups()]
        coords_np = np.asarray(coords).reshape(len(tile_file_list),2)
        coords_df = pd.DataFrame(coords_np, columns = ['x','y'])
        coords_df['labels'] = wsi_preds
        wsi_name = os.path.basename(wsi_file).split('.')[0]
        outfile_name = "./output/tcga-dlbc/coords_list_" + wsi_name + ".csv"
        coords_df.to_csv(outfile_name, index=False)

        # save file with annotations
        overlay_file = "./output/tcga-dlbc/overlay_" + wsi_name + '.png'
        save_heatmap(wsi_file, outfile_name, overlay_file)

    # save prediction info to csv
    preds_df = pd.DataFrame.from_dict(
        preds_dict, 
        orient='index',
        columns=['wsi_file','n_tumor','n_normal','t_purity'])
    print("preds_df:\n", preds_df)
    preds_df.to_csv(predict_csv, index=False)

    # perform rank correlation using WSI stage vs. WSI tumor purity
    rank_df = pd.DataFrame().assign(
        t_purity = preds_df['t_purity'], 
        stage = meta_df['stage'])
    print('rank_df:\n', rank_df)
    print(f'\n\tTotal rank correlation value is: {rank_df.corr()}')
