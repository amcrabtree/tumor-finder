import os
import geojson
import uuid
import glob
import matplotlib
from PIL import Image
import xml.etree.ElementTree as ET
import numpy as np
import openslide 
import glob
import cv2
import matplotlib.pyplot as plt

    
class AnnWSI (object):
    """
    Produces an annotated WSI object for use in tiling. 

    Data attributes:
       wsi: WSI image as OpenSlide object
       tumors: list of numpy arrays of tumor coordinates ([ [x1,x2,x3,...], [y1,y2,y3,...] ])
       
    Methods:
       constructor(v0): Set filenames of WSI (wsi_file)and annotations (ann_file).
       Resize(level, ratio, final_h, final_w): Print an example image from WSI.
       SaveOverlay(outfile): Check the appearance of how the annotation looks on a thumbnail image.
       StartTiling(outdir, size): Saves jpg images of tiles within descriptive (labeled) folders.

    Usage:
    >>> ann_wsi = AnnWSI (wsi_file, ann_file)
    """

    def __init__(self, wsi_file, roi_file:str=""):
        self.wsi_file = wsi_file
        self.roi_file = roi_file
        self.uuid = str(uuid.uuid1()).split('-')[0] # returns a short, unique (non-PHI) id for wsi 
        self.wsi = openslide.OpenSlide (wsi_file)
        self.tumors = self.LoadROIs()

    def LoadROIs (self) -> list:
        """ Parses annotation/ROI file into list of tumor ROI coordinates. """

        # define ROI filename and extension
        if self.roi_file == "": # if ROI filename is not provided, 
            roi_ext_list = ['.geojson', '.xml']
            for ex in roi_ext_list:
                pattern=os.path.join(
                    os.path.dirname(self.wsi_file),
                    os.path.basename(self.wsi_file).split(".")[0] + "*" + ex)
                match = "".join(glob.glob(pattern))
                if match != "": self.roi_file = match
        ext = os.path.splitext(os.path.abspath(self.roi_file))[1]

        # Parse XML file
        if ext == ".xml": 
            
            tree = ET.parse(self.roi_file)
            root = tree.getroot()

            # Make a list of all ROIs
            tumor_annotation=[el for el in root.findall('Annotations/Annotation')]

            # Make a list of Tumors
            tumors=[]
            for annotation in tumor_annotation:
                coords = np.array(
                    [ [float(el.attrib['X']),float(el.attrib['Y'])] for el in annotation.findall(
                        'Coordinates/Coordinate')]
                    )
                xy_coords = coords.flatten().reshape(int(coords.size/2), 2)
                tumors.append(xy_coords)

            #convert to list of np arrays
            tumors=[np.array(t) for t in tumors]
            return tumors

        # Parse GeoJSON file
        if ext == ".geojson":
            
            with open(self.roi_file) as f:
                gj = geojson.load(f)

                # Make a list of all ROIs
                tumor_annotation=[el for el in gj['features']]

                # Make a list of Tumors
                tumors=[]
                for annotation in tumor_annotation:
                    coords = np.array([el for el in annotation['geometry']['coordinates']], dtype=float)
                    xy_coords = coords.flatten().reshape(int(coords.size/2), 2)
                    tumors.append(xy_coords)

                # Return a list of np arrays
                tumors=[np.array(t) for t in tumors]
                return tumors






class Tile (AnnWSI):
    """ 
    Produces a tile object for use in tiling. 

    Data attributes:
        ann_obj: 
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

    def __init__(self, ann_obj:AnnWSI, id_coord:list, size:int, level:int=0): 
        
        self.ann_obj = ann_obj
        self.id_coord = np.array(id_coord)
        self.size = size
        self.level = level
        self.points = self.ReturnPoints()
        self.label = self.TileLabel()
        self.np_img = self.TileImg()
        self.blank = self.IsBlank()

    def ReturnPoints(self) -> np.ndarray:
        """ Returns level 0 xy coordinates of tile. """

        # calculate tile size at level zero
        base_tile_size = int(self.size * self.ann_obj.wsi.level_downsamples[self.level])
        
        # calculate point coordinates at level zero
        coords_list = [
            self.id_coord, 
            np.array([self.id_coord[0]+base_tile_size, self.id_coord[1]]), 
            np.array([self.id_coord[0]+base_tile_size, self.id_coord[1]+base_tile_size]), 
            np.array([self.id_coord[0], self.id_coord[1]+base_tile_size])
            ]
        points = np.asarray(coords_list).flatten().reshape(4,2)
        return points 

    def TileLabel(self) -> str:
        """ Returns label for tile depending if it's within any ROI. """

        # save coordinates as matplotlib Path object
        tumor_paths=[matplotlib.path.Path(t) for t in self.ann_obj.tumors]
        
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

    def TileImg(self) -> np.ndarray:
        """ Returns an image in a numpy array. """
        opens_img = self.ann_obj.wsi.read_region(
            (self.id_coord[0], self.id_coord[1]), 
            self.level, 
            (self.size,self.size))
        opens_img = opens_img.convert("RGB") # removing alpha channel
        np_img = np.array(opens_img)
        return np_img

    def IsBlank(self) -> bool:
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

    



def generate_tiles(ann_obj:AnnWSI, outdir:str,
                  tile_size:int, level:int, x_range:list=[], y_range:list=[]):
    """
    Iteratively generates and saves tiles over a specified area of a WSI.

    Arguments: 
        outdir: output directory to save tiles to
        ann_obj: the WSI represented as an AnnWSI object
        tile_size: the tile pixel dimensions (can only be square tiles)
        level: desired level of output tile images (0 is highest resolution)
        x_range: a 2-element list with [start, stop] coordinates of desired tiling area (default is WSI)
        y_range: a 2-element list with [start, stop] coordinates of desired tiling area (default is WSI)
    """
    
    # If no ranges are specified, tile whole slide 
    if x_range==[]:
        wsi_w, wsi_h = ann_obj.wsi.level_dimensions[0] # assuming range is level 0
        x_range = [0, wsi_w]
        y_range = [0, wsi_h]

    # Generate tiles
    n=1 # <- tile counter
    base_tile_size = int(tile_size * ann_obj.wsi.level_downsamples[level])
    for x in range(x_range[0], x_range[1]-base_tile_size, base_tile_size):
        for y in range(y_range[0], y_range[1]-base_tile_size, base_tile_size):

            # generate tile object and extract required info
            current_tile = Tile(ann_obj, [x,y], 256, level)

            # skip blank tiles
            if current_tile.blank == False:
            
                # save tile image to appropriate directory
                wsi_name=os.path.basename(ann_obj.wsi_file).split(".")[0]
                filename = f"wsi_{wsi_name}_tile_{n}_label_{current_tile.label}"
                outfile = os.path.join(outdir, filename)
                #np.save(outfile, current_tile.np_img)
                im = Image.fromarray(current_tile.np_img)
                im.save(outfile+".png")

                n += 1 # advance tile counter

    print(f"\n\tGenerated {n-1} tiles:\n")





def SaveOverlay(ann_obj:AnnWSI, mode:str, value, outfile:str=""):
    """ 
    Create image of WSI overlaid with annotations. 
    Adjust size with mode and value.

    Args:
        mode: attribute value quantifies ['ratio', 'level', 'height', 'width']
        value: value of mode
    
    Usage: 
    >>> updated_coords = ShrinkAnnCoords (ann_wsi, mode='height', value=1000)
    >>> updated_coords = ShrinkAnnCoords (ann_wsi, mode='width', value=1000)
    >>> updated_coords = ShrinkAnnCoords (ann_wsi, mode='level', value=2)
    >>> updated_coords = ShrinkAnnCoords (ann_wsi, mode='ratio' value=0.8)
    """
    # get wsi info and adjustment info
    wsi_w, wsi_h = ann_obj.wsi.level_dimensions[0]
    final_w, final_h = wsi_w, wsi_h # initialize final width & height

    # adjust sizing per arguments
    if mode == 'ratio':
        final_h = int(wsi_h*ratio)
        final_w = int(wsi_w*ratio)
    elif mode == 'height': # only height specified
        final_h = value
        ratio = final_h / wsi_h
        final_w = int(wsi_w*ratio)
    elif mode == 'width': # only width specified
        final_w = value
        ratio = final_w / wsi_w
        final_h = int(wsi_h*ratio)
    elif mode == 'level':
        final_w,final_h = ann_obj.wsi.level_dimensions[value]
    else: 
        print(f"Invalid mode selected. Please use: 'ratio', 'level', 'height', or 'width'")

    # transform input coordinates
    out_coords = []
    for t in ann_obj.tumors:
        x_arr, y_arr = t[:, 0], t[:, 1] # np arrays with all x and y values for one ROI
        new_x = np.round((x_arr * (final_w / wsi_w)), 0)
        new_y = np.round((y_arr * (final_h / wsi_h)), 0)
        xy_coords = np.array([new_x, new_y], dtype=float).flatten('F').reshape(int(x_arr.size), 2)
        out_coords.append(xy_coords)

    # resize WSI
    level = 4 # using level 4 so it takes less time to load (more pixelated though)
    wsi = ann_obj.wsi.read_region((0, 0), level, ann_obj.wsi.level_dimensions[level])
    wsi = wsi.convert("RGB") # removing alpha channel
    wsi_np = np.array(wsi)
    shrunken_img = cv2.resize(wsi_np, dsize=(final_w, final_h), interpolation=cv2.INTER_CUBIC)
    
    # show or save image
    plt.imshow(shrunken_img)
    for t in out_coords: plt.scatter(t[:, 0], t[:, 1], s=10, marker='.', c='g')
    if outfile == "": # show image popup if no output file is specified
        plt.show()
    else: 
        print(f"\n\tThere are {len(out_coords)} annotated tumor regions.")
        plt.gcf().set_size_inches(20, 20)
        plt.savefig(outfile, dpi=100)
        print(f"\n\tSaved overlay to: {outfile}\n")
