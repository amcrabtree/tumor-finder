#!/usr/bin/env python3

import os
import sys
import glob
from tumor_utils.tiling import AnnWSI
import tumor_utils.tiling
import tumor_utils.mv_tiles
import json

"""
Generates tiles for all WSIs in WSI folder.
"""

if __name__ == "__main__":

    # initial run configurations 
    config_file = open(sys.argv[1], "r")
    config = json.loads(config_file.read())
    config_file.close()
    
    WSI_DIR=config['wsi_dir']
    TILE_DIR=config['tile_dir']
    for dir in [WSI_DIR, TILE_DIR]:
        if not os.path.exists(dir):
            print(f"Directory does not exist! Try again. \n\t{dir}")
            exit(1)
    
    for wsi_file in glob.glob(WSI_DIR+"/*"):

        ext_list = ['.ndpi', '.tiff', '.tif']
        wsi_ext = os.path.splitext(wsi_file)[1]
        if wsi_ext in ext_list:

            # create annotated WSI object
            ann_wsi = AnnWSI (wsi_file)

            # check overlay of annotations on WSI
            # thumbnail_file = os.path.splitext(wsi_file)[0]+"_thumb.jpg"
            # tumor_utils.tiling.SaveOverlay(
            #     ann_wsi, mode="height", value=10000) 

            # generate tiles
            print(f"\nTiling file: {os.path.basename(wsi_file)}")
            tumor_utils.tiling.GenerateTiles (
                ann_wsi, outdir=TILE_DIR,
                tile_size=256,
                level=0)

    # subdivide tiles into training, validation, and testing folders
    tumor_utils.mv_tiles.balance_classes(TILE_DIR, sm_class='tumor', lg_class='normal', ratio=0.5) # balance 2 classes
    tumor_utils.mv_tiles.split_tile_dirs (TILE_DIR, ratio=0.2)
    tumor_utils.mv_tiles.rm_class_dir(TILE_DIR, "margin") # remove margin tile folders

    print(f"\n\nSaved all resultant tiles within:  \n\t{os.path.abspath(TILE_DIR)}\n")
