
import pandas as pd
import torch
from tumor_utils import tformer # custom class for loading & transforming data
from tumor_utils.data import UnlabeledImgData # custom class for unlabeled images
from tumor_utils.tiling import tile_wsi_list # custom tiling function
from tumor_utils.prediction import TumorPredWSI # custom tumor prediction class


if __name__=='__main__':

    #wsi_dir="/projects/bgmp/acrabtre/tumor-finder/data/tcga"
    wsi_dir="/projects/bgmp/acrabtre/tumor-finder/data/wsi/prod_wsi"
    wsi_ext=".tif" # on TCGA files will be ".svs"
    meta_file="/projects/bgmp/acrabtre/tumor-finder/data/wsi/prod_wsi/meta_prod.txt"
    #tile_dir="/projects/bgmp/acrabtre/tumor-finder/data/tcga/tiles"
    tile_dir="/projects/bgmp/acrabtre/tumor-finder/data/tiles/prod"
    model_file="/projects/bgmp/acrabtre/tumor-finder/output/WSI_VGG_Adam/running_model.pt"
    tiling=False

    # import metadata as pandas df and save file list
    meta_df = pd.read_csv(meta_file, sep='\t')
    meta_df['pathname'] = wsi_dir + "/" + meta_df['slide_id'] + wsi_ext
    wsi_file_list = [x for x in meta_df['pathname']]
    wsi_stage_list = [x for x in meta_df['stage']]

    # tile wsis if needed
    if tiling==True: tile_wsi_list(wsi_file_list, tile_dir, tile_size=256, level=0)

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
        orient='index',
        columns=['wsi_file','n_tumor','n_normal','t_purity'])
    print("preds_df:\n", preds_df)

    # perform rank correlation using WSI stage vs. WSI tumor purity
    #rank_df = pd.concat([meta_df, preds_df], ignore_index=True, sort=False)
    rank_df = pd.DataFrame().assign(
        t_purity = preds_df['t_purity'], 
        stage = meta_df['stage'])
    print('rank_df:\n', rank_df)
    print(f'\n\tTotal rank correlation value is: {rank_df.corr()}')
