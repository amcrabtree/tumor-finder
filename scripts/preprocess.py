#!/usr/bin/env python3

import os
import sys
import glob
from tumor_utils.tiling import AnnWSI
import tumor_utils.tiling
import tumor_utils.mv_tiles
import json
from tumor_utils.wsi import WSI # custom WSI class

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
            ann_wsi = WSI(wsi_file)
            # check overlay of annotations on WSI
            # WSI.save_overlay(ann_wsi, mode="height", value=10000) 

            # generate tiles
            print(f"\nTiling file: {os.path.basename(wsi_file)}")
            WSI.generate_tiles(
                ann_wsi, outdir=TILE_DIR,
                tile_size=256,
                level=0)
    
    # balance 2 classes and move unneeded files to 'excess' directory
    tumor_utils.mv_tiles.balance_classes(TILE_DIR, class_a='tumor', class_b='normal', ratio=1) 
    # subdivide tiles into training, validation, and testing directories
    tumor_utils.mv_tiles.split_tile_dirs(TILE_DIR, ratio=0.2)
    # remove margin tile folders
    tumor_utils.mv_tiles.rm_class_dir(TILE_DIR, "margin") 

    print(f"\n\nSaved all resultant tiles within:  \n\t{os.path.abspath(TILE_DIR)}\n")
