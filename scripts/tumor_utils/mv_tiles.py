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

def balance_numbers(n_class_a:int, n_class_b:int, ratio:float)->tuple:
    """ Returns tuple of numbers of classes a and b to fulfil ratio.

    Args:
        n_class_a: number of items in class_a
        n_class_b: number of items in class_b
        ratio: (# images in class_a) / (# images in class_b)
    """
    n_a_out, n_b_out = 0,0
    if n_class_a > n_class_b*ratio:
        n_a_out, n_b_out = round(n_class_b * ratio), n_class_b
    elif n_class_a < n_class_b*ratio:
        n_a_out, n_b_out = n_class_a, round(n_class_a / ratio)
    else:
        n_a_out, n_b_out = n_class_a, n_class_b
    return n_a_out, n_b_out

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

def balance_classes (tile_dir:str, class_a:str, class_b:str, ratio:float=1):
    """ Balances 2 classes based on class labels and ratio.
    Ex: if classes are ['tumor', 'normal'] with 100 and 100 tiles respectively
    and ratio=0.5, the new balance will be 50 tumor files and 100 normal, 
    using maximum number of tumor files available for each wsi. 

    Args:
        tile_dir: directory containing tile images
        class_a: the label class A 
        class_b: the label class B
        ratio: desired ratio of class A to class B (ratio = A / B)
    Output:
        (none) moves all excess files to 'excess' directory
    """
    # make nested dict of WSI names with labels: file counts
    wsi_dict = {} # keys=WSI names, values=dict of counts for each class label
    for f in glob.glob(tile_dir+"/*.*"):
        match = re.search('wsi_(\w+)_tile_.+_label_(\w+)\..+', os.path.basename(f))
        f_wsi, f_label = match.group(1), match.group(2)
        if f_wsi not in wsi_dict.keys(): 
            wsi_dict[f_wsi]={class_a:0, class_b:0} # keys=labels, values=list of file counts
        if f_label in [class_a, class_b]:
            wsi_dict[f_wsi][f_label] += 1

    # create 'excess' directory
    dest_dir=os.path.join(tile_dir, 'excess')
    if not os.path.exists(dest_dir): # make destination dir if needed
            os.makedirs(dest_dir)

    # move number of random images of excess class to 'excess' dir
    print(f"\nMoving excess tiles...\n")
    for wsi in wsi_dict:
        rm_list = [] # list of files to remove
        n_class_a = wsi_dict[wsi][class_a]
        n_class_b = wsi_dict[wsi][class_b]
        n_a_out, n_b_out = balance_numbers(n_class_a, n_class_b, ratio)
        if n_a_out < n_class_a:
            n_imgs_rm = n_class_a - n_a_out
            rm_list = find_rand_n_files(tile_dir, match_list=[wsi, class_a], n_files=n_imgs_rm)
        if n_b_out < n_class_b:
            n_imgs_rm = n_class_b - n_b_out
            rm_list = find_rand_n_files(tile_dir, match_list=[wsi, class_b], n_files=n_imgs_rm)
        # remove excess files
        for f in rm_list:
            f_basename = os.path.basename(f)
            shutil.move(f, os.path.join(dest_dir, f_basename))
        print(f"WSI: {wsi}")
        print(f"\tn {class_a} tiles = {n_a_out}")
        print(f"\tn {class_b} tiles = {n_b_out}")




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

def reset_tiles(tile_dir:str):
    """ Move all images from subfolders back to main tile directory. """
    # move images
    for f in glob.glob(tile_dir+"/**/*.*", recursive=True):
        f_dest = os.path.join(tile_dir, os.path.basename(f))
        shutil.move(f, f_dest)
    # remove old directories
    for d in glob.glob(tile_dir+"/**/"):
        print("removing:", d)
        shutil.rmtree(d, ignore_errors=True)



if __name__ == "__main__":

    TILE_DIR=sys.argv[1]
    balance_classes(TILE_DIR, sm_class='tumor', lg_class='normal', ratio=0.5) # balance 2 classes
    split_tile_dirs(TILE_DIR, ratio=0.2) # move tiles to subdirs
    rm_class_dir (TILE_DIR, "margin") # remove any margin folders
