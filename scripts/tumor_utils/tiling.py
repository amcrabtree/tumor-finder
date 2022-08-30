import os
from tumor_utils.wsi import WSI # custom WSI class
from tumor_utils.tile import Tile # custom Tile class

    
def tile_wsi_list(wsi_file_list:list, tile_dir:str, tile_size:int=256, level:int=0):
    # generate a WSI object for each WSI in list
    for wsi_file in wsi_file_list:
        print(f"\nTiling file: {os.path.basename(wsi_file)}")
        wsi_obj = WSI(wsi_file)
        coods_list = wsi_obj.get_tile_coords(tile_size, level)
        # Generate tiles from WSI object
        n=1 # <- tile counter
        for x,y in coods_list:
            # generate tile object and save all non-blank tiles
            current_tile = Tile(wsi_obj, [x,y], tile_size, level)
            current_tile.save_tile(tile_dir, n, ext=".png")
            n += 1 
            