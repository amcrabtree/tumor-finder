import sys
import os
import random
import re
import glob
from math import ceil
import shutil

# If run as __main__: divides tiles for all WSIs in appropriate folders and gets rid of margin tiles.

def SplitTileDirs (tile_dir:str):
    """ Splits tile images into Training and Validation directories.
    """
    # make sets of WSI names & unique labels
    wsi_set = set() # set of WSI filenames
    label_set = set() # set of labels
    for f in glob.glob(tile_dir+"/*.*"):
        match = re.search('wsi_(\w+)_tile_.+_label_(\w+)\..+', os.path.basename(f))
        f_wsi, f_label = match.group(1), match.group(2)
        if f_wsi not in wsi_set: wsi_set.add(f_wsi)
        if f_label not in label_set: label_set.add(f_label)

    # split WSIs into train/val/test
    def SplitStringList(my_list:list[str], ratio:float=0.1) -> list:
        """ Divides a string list into 3 lists of shuffled elements. 
        Ex: ratio of 0.1 means 10% test, 10% val, 80% train 
        """
        if ratio > len(my_list)//3:
            print("Ratio must be less than 1/3 of the length of list. Try again.")
            exit(1)
        random.shuffle(my_list) # shuffle list elements
        n_items = ceil(len(my_list)*ratio)
        test_list = my_list[:n_items]
        remaining_list = my_list[n_items:]
        val_list = remaining_list[:n_items]
        train_list = remaining_list[n_items:]
        return [train_list, val_list, test_list]
    
    train_wsis, val_wsis, test_wsis = SplitStringList(list(wsi_set))
    wsi_dict = {'train':train_wsis, 'val':val_wsis, 'test':test_wsis}
    print(wsi_dict)

    # move tile files to destination directories
    for f in glob.glob(tile_dir+"/*.*"):
        match = re.search('wsi_(\w+)_tile_.+_label_(\w+)\..+', os.path.basename(f))
        wsi, label = match.group(1), match.group(2)
        split_set = "".join([k for k, v in wsi_dict.items() if wsi in v])
        dest_dir = os.path.join(tile_dir, split_set, label)
        if not os.path.exists(dest_dir): # make destination dir if needed
            os.makedirs(dest_dir)
        output_filename = os.path.join(dest_dir, os.path.basename(f))
        os.replace(f, output_filename)


def RemoveClassDir (tile_dir:str, label_dir:str):
    """ Removes unwanted folder, label_dir, from all directories in tile_dir. """
    for split_dir in glob.glob(tile_dir+"/*"):
        unwanted_path = os.path.join(split_dir, label_dir)
        shutil.rmtree(unwanted_path, ignore_errors=True)


if __name__ == "__main__":

    TILE_DIR=sys.argv[1]
    SplitTileDirs(TILE_DIR, ratio=0.2) # move tiles to subdirs
    RemoveClassDir (TILE_DIR, "margin") # remove any margin folders
