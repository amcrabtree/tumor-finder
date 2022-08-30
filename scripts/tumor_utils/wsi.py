import os
import geojson
import uuid
import glob
import xml.etree.ElementTree as ET
import numpy as np
import openslide 
import glob
import cv2
import matplotlib.pyplot as plt


class WSI (object):
    """
    Produces a WSI object for use in tiling. 

    Data attributes:
       wsi: WSI image as OpenSlide object
       tumors: list of numpy arrays of tumor coordinates ([ [x1,x2,x3,...], [y1,y2,y3,...] ])
       
    Methods:
       constructor(v0): Set filenames of WSI (wsi_file)and annotations (ann_file).
       load_roi: Parses annotation/ROI file into list of tumor ROI coordinates.
       save_overlay: Create image of WSI overlaid with annotations. 

    Usage:
    >>> ann_wsi = WSI (wsi_file, ann_file)
    >>> unlabeled_wsi = WSI (wsi_file)
    """

    def __init__(self, wsi_file, roi_file:str=""):
        self.wsi_file = wsi_file
        self.roi_file = roi_file
        self.uuid = str(uuid.uuid1()).split('-')[0] # returns a short, unique (non-PHI) id for wsi 
        self.wsi = openslide.OpenSlide (wsi_file)
        self.tumors = [] if roi_file=="" else self.load_roi()

    def load_roi (self) -> list:
        """ Parses annotation/ROI file into list of tumor ROI coordinates. 
        """
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

    def save_overlay(self, mode:str, value, outfile:str=""):
        """ 
        Create image of WSI overlaid with annotations. 
        Adjust size with mode and value.

        Args:
            mode: attribute value quantifies ['ratio', 'level', 'height', 'width']
            value: value of mode
        Opts:
            outfile: where the overlay will be saved to
        
        Usage: 
        >>> updated_coords = ShrinkAnnCoords (ann_wsi, mode='height', value=1000)
        >>> updated_coords = ShrinkAnnCoords (ann_wsi, mode='width', value=1000)
        >>> updated_coords = ShrinkAnnCoords (ann_wsi, mode='level', value=2)
        >>> updated_coords = ShrinkAnnCoords (ann_wsi, mode='ratio' value=0.8)
        """
        # handle opts
        if outfile=="": outfile=os.path.splitext(self.wsi_file)[0]+"_thumb.jpg"

        # get wsi info and adjustment info
        wsi_w, wsi_h = self.wsi.level_dimensions[0]
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
            final_w,final_h = self.wsi.level_dimensions[value]
        else: 
            print(f"Invalid mode selected. Please use: 'ratio', 'level', 'height', or 'width'")

        # transform input coordinates
        out_coords = []
        for t in self.tumors:
            x_arr, y_arr = t[:, 0], t[:, 1] # np arrays with all x and y values for one ROI
            new_x = np.round((x_arr * (final_w / wsi_w)), 0)
            new_y = np.round((y_arr * (final_h / wsi_h)), 0)
            xy_coords = np.array([new_x, new_y], dtype=float).flatten('F').reshape(int(x_arr.size), 2)
            out_coords.append(xy_coords)

        # resize WSI
        level = 4 # using level 4 so it takes less time to load (more pixelated though)
        wsi = self.wsi.read_region((0, 0), level, self.wsi.level_dimensions[level])
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

    def get_tile_coords(
        self, tile_size:int, level:int, x_range:list=[], y_range:list=[])->np.ndarray:
        """ Iteratively generates tiles coordinates over a specified area of a WSI.

        Args: 
            tile_size: the tile pixel dimensions (can only be square tiles)
            level: desired level of output tile images (0 is highest resolution)

        Opts:
            x_range: a 2-element list with [start, stop] coordinates of desired tiling area (default is WSI)
            y_range: a 2-element list with [start, stop] coordinates of desired tiling area (default is WSI)
        
        Usage:
        >>> get_tile_coords(tile_size=256, level=0)
        >>> get_tile_coords(tile_size=256, level=0,
            ... x_range=[13000,16000], y_range=[4000,8000])
        """
        # If no ranges are specified, tile whole slide 
        if x_range==[]:
            wsi_w, wsi_h = self.wsi.level_dimensions[0] # assuming range is level 0
            x_range = [0, wsi_w]
            y_range = [0, wsi_h]

        # Generate tile coords
        tile_coord_list = []
        base_tile_size = int(tile_size * self.wsi.level_downsamples[level])
        for x in range(x_range[0], x_range[1]-base_tile_size, base_tile_size):
            for y in range(y_range[0], y_range[1]-base_tile_size, base_tile_size):
                tile_coord_list.append(np.array([x,y]))
        return np.asarray(tile_coord_list)
