import os
import matplotlib
from PIL import Image
import numpy as np
import glob
from wsi import WSI # custom WSI class

class Tile (WSI):
    """ 
    Produces a tile object for use in tiling. 

    Data attributes:
        wsi_obj: WSI class object
        id_coord: top left tile coordinate (in image coordinate system)
        size: desired output tile size in pixels (h=w)
        level: desired output level (0 is highest quality level)
        points: list of the 4 xy coordinates comprising tile corners at level 0
        label: labels tile according to ROIs ('tumor', 'normal' or 'margin')
        np_img: numpy array of tile image 
       
    Methods:
        get_corners(): Return list of corner point coordinates, adjusted to level 0 coordinates
        get_tile_label(): Returns tile label, according to annotated ROIs.
        get_np_img(): Converts tile to numpy array.
        save_tile(): Save tile to file

    Usage:
    >>> new_tile = Tile(ann_wsi, id_coord=[4600,15000], size=96, level=4)
    >>> new_tile = Tile(unlabeled_wsi, id_coord=[4600,15000], size=256, level=0)
    """

    def __init__(self, wsi_obj:WSI, id_coord:list, size:int, level:int=0): 
        self.wsi_obj = wsi_obj
        self.id_coord = np.array(id_coord)
        self.size = size
        self.level = level
        self.np_img = self.get_np_img()
        self.blank = self.is_blank_tile()
        if wsi_obj.roi_file != "":
            self.points = self.get_corners()
            self.label = self.get_tile_label()

    def get_corners(self) -> np.ndarray:
        """ Returns level 0 xy coordinates of tile. """

        # calculate tile size at level zero
        base_tile_size = int(self.size * self.wsi_obj.wsi.level_downsamples[self.level])
        
        # calculate point coordinates at level zero
        coords_list = [
            self.id_coord, 
            np.array([self.id_coord[0]+base_tile_size, self.id_coord[1]]), 
            np.array([self.id_coord[0]+base_tile_size, self.id_coord[1]+base_tile_size]), 
            np.array([self.id_coord[0], self.id_coord[1]+base_tile_size])
            ]
        points = np.asarray(coords_list).flatten().reshape(4,2)
        return points 

    def get_tile_label(self) -> str:
        """ Returns label for tile depending if it's within any ROI. """

        # save coordinates as matplotlib Path object
        tumor_paths=[matplotlib.path.Path(t) for t in self.wsi_obj.tumors]
        
        # list: length is num ROIs, values are num tile points/corners inside ROI
        sum_list = [sum(path.contains_points(self.points).tolist()) for path in tumor_paths]
        
        # define tile label
        label=""
        if sum_list.count(4) > 0:   # in at least 1 ROI
            label="tumor"
        elif sum_list.count(0) == len(sum_list):   # outside all ROIs
            label="normal"
        else:                       
            label="margin" 
        return label

    def get_np_img(self) -> np.ndarray:
        """ Returns an image in a numpy array. """
        opens_img = self.wsi_obj.wsi.read_region(
            (self.id_coord[0], self.id_coord[1]), 
            self.level, 
            (self.size,self.size))
        opens_img = opens_img.convert("RGB") # removing alpha channel
        np_img = np.array(opens_img)
        return np_img

    def is_blank_tile(self) -> bool:
        """ Returns True if tile is blank. """
        is_blank=False # assume tile is not blank (default)
        # convert np array to B&W array using relative luminance
        def rgb2lum(rgb):
            r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
            gray = 0.2126 * r + 0.7152 * g + 0.0722 * b
            return gray
        lum_img = rgb2lum(self.np_img)

        # if tiles are all black (in the case of PCam WSI blank space):
        ratio_whitish = np.count_nonzero(lum_img > 180) / lum_img.size
        if ratio_whitish > 0.8: is_blank=True

        # if tiles are mostly white (in the case of normal WSI blank space)
        ratio_blackish = np.count_nonzero(lum_img < 50) / lum_img.size
        if ratio_blackish > 0.8: is_blank=True 
        if is_blank: self.label = "blank"
        return is_blank

    def save_tile(self, outdir:str="", n:int=-1, ext=".png"):
        """ Save tile to file
        Options:
            outdir: output directory for tile image (default is WSI dir)
            n: tile number/id included in filename (default is tile coords)
            ext: image extension (".png", ".jpg", or ".np")

        """
        # handle options
        if outdir=="": outdir=self.wsi_obj.wsi_file
        if n==-1: n=='{}-{}'.format(self.id_coord[0],self.id_coord[1])
        # save tile image to appropriate directory
        wsi_name=os.path.basename(self.wsi_obj.wsi_file).split(".")[0]
        filename = f"wsi_{wsi_name}_tile_{n}_label_{self.label}"
        outfile = os.path.join(outdir, filename)
        if ext==".np":
            np.save(outfile, self.np_img)
        elif ext==(".png" or ".jpg" or ".jpeg"):
            im = Image.fromarray(self.np_img)
            im.save(outfile+ext)
        else: 
            print('ERROR: extension not allowed. Please change to ".png", ".jpg", or ".np"')
            exit(1)
