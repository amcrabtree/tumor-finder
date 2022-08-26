
import numpy as np
import pandas as pd
from tumor_utils.tiling import tile_unlabeled_wsi
from PIL import Image

wsi_dir="/projects/bgmp/acrabtre/tumor-finder/data/tcga"
meta_file="/projects/bgmp/acrabtre/tumor-finder/data/tcga/tcga_meta.csv"
tile_dir="/projects/bgmp/acrabtre/tumor-finder/data/tcga/tiles"
model_file=""
# import metadata as pandas df
df = pd.read_csv(meta_file)
#wsi_dict_list = df.to_dict('records') # key will be index (an int)

# iterate through WSIs/slides/rows
for index, wsi in df.iterrows():
    #print(wsi['biopsy_id'], wsi['stage'])
    wsi_file = wsi['slide_id'] + ".svs"

    # tile wsi
    tile_unlabeled_wsi(wsi_file, tile_dir=tile_dir)

    # run model on all its tiles and save numbers of normal vs. tumor tiles
    preds_dict = TilePreds(tile_dir=tile_dir, model_file=model_file) 


# perform rank correlation using WSI stage vs. WSI tumor purity
rank_corr()


class TilePreds():
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
    def __init__(self, tile_dir:str, size:int=256, level:int=0):
        self.tile_dir = tile_dir
        self.size = size
        self.level = level
        self.points = self.get_points()
        self.np_img = self.get_tile_img()

    def get_tile_img(self) -> np.ndarray:
        """ Returns a tile image as a numpy array. """
        im = Image.open(self.tile) 
        np_img = np.array(im)
        return np_img

