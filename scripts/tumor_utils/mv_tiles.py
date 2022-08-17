#!/usr/bin/env python3

import sys
import os
import random
import re
import glob
from math import ceil
import shutil

"""If run as __main__: divides tiles for all WSIs in appropriate folders and gets rid of margin tiles.
"""

def balance_classes (tile_dir:str, sm_class:str, lg_class:str, ratio:float=0.5):
    """ Balances 2 classes based on class labels and ratio.
    Ex: if classes are ['tumor', 'normal'] and ratio=0.3, 
    the new balance will be 30% tumor files and 70% normal, 
    using maximum number of tumor files available for each wsi. 

    Args:
        tile_dir: directory containing tile images
        sm_class: the label of the smaller class used for reference
        lg_class: the label of the larger class used for thinning
        ratio: desired ratio of small class to large class 
    Output:
        (none) moves all extra lg_class files to 'excess' directory
    """
    # make nested dict of WSI names with labels: file counts
    wsi_dict = {} # keys=WSI names, values=dict of counts for each class label
    for f in glob.glob(tile_dir+"/*.*"):
        match = re.search('wsi_(\w+)_tile_.+_label_(\w+)\..+', os.path.basename(f))
        f_wsi, f_label = match.group(1), match.group(2)
        if f_wsi not in wsi_dict.keys(): 
            wsi_dict[f_wsi]={sm_class:0, lg_class:0} # keys=labels, values=list of file counts
        if f_label in [sm_class, lg_class]:
            wsi_dict[f_wsi][f_label] += 1
    # create 'excess' directory

    # move number of random images of lg_class to 'excess' dir
    def find_rand_n_files (dir:str, match_list:list, n_files:int) ->list:
        """ Return list of n number of filenames matching all search strings. 
        Args:
            dir: file directory to search filenames
            match_list: list of strings to match (filename must match all)
            n_files: number of file matches to return (will be random set of all matches)
        Output:
            list of n number of filenames matching all search strings
        """
        # make list of all matches
        file_matches=[]
        for f in glob.glob(dir+"/*.*"):
            if all(m in f for m in match_list):
                file_matches.append(f)
        # retain only n_files number of random images from list
        random.shuffle(file_matches)
        file_matches=file_matches[:n_files]
        return file_matches

    print(f"\nMoving excess tiles...\n")
    dest_dir=os.path.join(tile_dir, 'excess')
    if not os.path.exists(dest_dir): # make destination dir if needed
            os.makedirs(dest_dir)

    for wsi in wsi_dict:
        n_sm_imgs = wsi_dict[wsi][sm_class]
        n_lg_imgs = wsi_dict[wsi][lg_class]
        n_lg_imgs_rm = int(n_lg_imgs - (((1-ratio) * n_sm_imgs) // ratio))
        rm_list = find_rand_n_files(tile_dir, match_list=[wsi, lg_class], n_files=n_lg_imgs_rm)
        # remove excess files
        for f in rm_list:
            f_basename = os.path.basename(f)
            shutil.move(f, os.path.join(dest_dir, f_basename))
        print(f"WSI: {wsi}")
        print(f"\tn {sm_class} tiles = {n_sm_imgs}")
        print(f"\tn {lg_class} tiles = {n_lg_imgs-n_lg_imgs_rm}")




def split_tile_dirs (tile_dir:str, ratio:float=0.2):
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
    def split_string_list(my_list:list[str], ratio:float=0.2) -> list:
        """ Divides a string list into 3 lists of shuffled elements. 
        Ex: ratio of 0.2 means 20% test, 20% val, 60% train 
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
    
    train_wsis, val_wsis, test_wsis = split_string_list(list(wsi_set), ratio)
    wsi_dict = {'train':train_wsis, 'val':val_wsis, 'test':test_wsis}

    # move tile files to destination directories
    print(f"\nMoving tiles to trail/val/test directories...\n")
    for k in wsi_dict: print(f"\t{k}: {wsi_dict[k]}")
    for f in glob.glob(tile_dir+"/*.*"):
        match = re.search('wsi_(\w+)_tile_.+_label_(\w+)\..+', os.path.basename(f))
        wsi, label = match.group(1), match.group(2)
        split_set = "".join([k for k, v in wsi_dict.items() if wsi in v])
        dest_dir = os.path.join(tile_dir, split_set, label)
        if not os.path.exists(dest_dir): # make destination dir if needed
            os.makedirs(dest_dir)
        output_filename = os.path.join(dest_dir, os.path.basename(f))
        os.replace(f, output_filename)


def rm_class_dir (tile_dir:str, label_dir:str):
    """ Removes unwanted folder, label_dir, from all directories in tile_dir. """
    for split_dir in glob.glob(tile_dir+"/*"):
        unwanted_path = os.path.join(split_dir, label_dir)
        shutil.rmtree(unwanted_path, ignore_errors=True)


if __name__ == "__main__":

    TILE_DIR=sys.argv[1]
    balance_classes(TILE_DIR, sm_class='tumor', lg_class='normal', ratio=0.5) # balance 2 classes
    split_tile_dirs(TILE_DIR, ratio=0.2) # move tiles to subdirs
    rm_class_dir (TILE_DIR, "margin") # remove any margin folders