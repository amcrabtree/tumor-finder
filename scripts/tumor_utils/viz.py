import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns; sns.set_theme()
from torch.utils.data import Dataset
import random
import torch
import pandas as pd
from plotnine import *
import cv2
import openslide 
from scipy.stats import gaussian_kde
import os

def ViewNpyImg(npy_file:str):
    """ Display .npy tile image as pop-up. """
    tile_np = np.load(npy_file)
    plt.imshow(tile_np, interpolation='nearest')
    plt.show()

def print_sample_imgs(dataset:Dataset, outfile:str):
    """ Print to file sample images and labels. 
    """
    fig = plt.figure(figsize=(12,4))
    sample_list = random.sample(range(len(dataset)), 5)
    for i, idx in enumerate(sample_list):
        image, label = dataset[idx]
        if type(image) == torch.Tensor:
            image = image.permute(1, 2, 0).numpy()
            image = np.clip(image, 0, 1)
        label = "normal" if label==0 else "tumor"
        #print(idx, image.shape, type(image), label)
        plt.figure(1)
        plt.subplot(1,5,i+1)
        plt.gca().set_title(label)
        plt.axis('off')
        plt.imshow(image)
    plt.show()
    plt.savefig(outfile)

def plot_loss(df:pd.DataFrame, outfile:str=""):
    # line plot: epoch loss
    p = (ggplot(df) +
        aes(x='epoch', y='epoch_loss', group='phase') + 
        geom_line(aes(color='phase'))+ 
        geom_point(aes(color='phase')) + 
        theme_classic() + 
        labs(title="Epoch Loss", x="Epoch", y="Loss") 
    )
    # save file
    filename = "plot_loss.png" if outfile == "" else outfile
    p.save(filename=filename,
        plot=p,
        device='png',
        dpi=300,
        height=3,
        width=6,
        verbose = False)


def plot_acc(df:pd.DataFrame, outfile:str=""):
    # line plot: epoch accuracy
    p = (ggplot(df) +
        aes(x='epoch', y='epoch_acc', group='phase') + 
        geom_line(aes(color='phase')) + 
        geom_point(aes(color='phase')) + 
        theme_classic() + 
        labs(title="Epoch Accuracy", x="Epoch", y="Accuracy") + 
        ylim(0, 1)
    )
    # save file
    filename = "plot_acc.png" if outfile == "" else outfile
    p.save(filename=filename,
        plot=p,
        device='png',
        dpi=300,
        height=3,
        width=6,
        verbose = False)

def confusion_plot(matrix:np.ndarray, outfile:str=""):
    """ Plot confusion matrix for classification model.
    """
    class_labels = ['normal','tumor']
    sns.heatmap(
        matrix, annot=True, fmt="d", cmap="Blues", 
        xticklabels=class_labels, yticklabels=class_labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Prediction')
    plt.ylabel('Ground Truth')
    plt.show()
    filename = "plot_confusion.png" if outfile == "" else outfile
    plt.savefig(filename)
    print('Number of Correct Predictions:',np.trace(matrix))
    print('Number of Incorrect Predictions:',np.fliplr(matrix).trace())
    print('Number of Total Predictions:', np.sum(matrix))

def roc_plot(df:pd.DataFrame, auroc=0, outfile:str=""):
    """ Plot ROC curve. 
    Args:
        df: pandas df with the columns = [model,fpr,tpr,thresholds]
    """
    title = "ROC Curve" if auroc==0 else "ROC Curve (AUC={})".format(auroc)
    p = (ggplot(df) +
        aes(x='fpr', y='tpr') + 
        #geom_area(fill="#69b3a2", alpha=0.4) + # for some reason, looks messed up
        geom_line(color="#69b3a2", size=2) +
        theme_classic() +
        labs(title=title, x="False Positive Rate (specificity)", y="True Positive Rate (sensitivity)") + 
        ylim(0, 1) + 
        xlim(0, 1)
    )
    # save file
    filename = "plot_roc.png" if outfile == "" else outfile
    p.save(filename=filename,
        plot=p,
        device='png',
        dpi=300,
        height=4,
        width=6,
        verbose = False)

def save_heatmap(wsi_file:str, coord_file:str, outfile:str=""):
    """ 
    Create image of WSI overlaid with tumor heatmap. 

    Args:
        wsi_file: path of wsi image
        coord_file: path of file containing x,y coordinates
        outfile: name of output file
    
    Usage: 
    >>> save_heatmap ("my_wsi.ndpi", "coords.txt")
    >>> save_heatmap ("my_wsi.ndpi", "coords.txt", "overlay.png")
    """
    # get wsi info and adjustment info
    input_wsi = openslide.OpenSlide (wsi_file)
    wsi_w, wsi_h = input_wsi.level_dimensions[0]
    final_w, final_h = wsi_w, wsi_h # initialize final width & height

    # adjust sizing per desired pixel height 
    final_h = 10000
    ratio = final_h / wsi_h
    final_w = int(wsi_w*ratio)

    # resize WSI
    level = 2 # using level 2 so it takes less time to load (more pixelated though)
    wsi = input_wsi.read_region((0, 0), level, input_wsi.level_dimensions[level])
    wsi = wsi.convert("RGB") # removing alpha channel
    wsi_np = np.array(wsi)
    shrunken_img = cv2.resize(wsi_np, dsize=(final_w, final_h), interpolation=cv2.INTER_CUBIC)
    
    # transform input coordinates
    delim=',' if os.path.splitext(coord_file)[-1]=='.csv' else '\t'
    tumor_np = np.genfromtxt(coord_file, delimiter=delim)
    tumor_list = [c for c in tumor_np if not np.isnan(c[0])] # removes headers
    adj_coords = []
    for t in tumor_list:
        x, y, label = t[0], t[1], t[2] # np arrays with all x and y values for one tile + label
        new_x = (x * (final_w / wsi_w))
        new_y = (y * (final_h / wsi_h))
        xy_coords = np.array([new_x, new_y], dtype=float)
        if label==1: adj_coords.append(xy_coords)
    x = np.asarray([c[0] for c in adj_coords])
    y = np.asarray([c[1] for c in adj_coords])
    #z = np.ones([len(adj_coords),len(adj_coords)])

    # Evaluate a gaussian kde on a regular grid of nbins x nbins over data extents
    nbins=300
    k = gaussian_kde([x,y])
    xi, yi = np.mgrid[0:final_w:nbins*1j, 0:final_h:nbins*1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))
    
    # plot (for small image without overlay, alpha=0)
    fig=plt.figure()
    plt.imshow(shrunken_img)
    plt.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='auto', cmap=plt.cm.Reds, alpha=0.5)

    # show or save resulting image
    if outfile == "": # show image popup if no output file is specified
        plt.show()
    else: 
        print(f"\n\tThere were {len(adj_coords)} tumor tiles.")
        fig.set_size_inches(final_h/100, final_w/100)
        fig.savefig(outfile, dpi=100)
        print(f"\n\tSaved overlay to: {outfile}\n")
        plt.close(fig)

if __name__=='__main__':

    wsi_file = "/projects/bgmp/acrabtre/tumor-finder/data/wsi/prod_wsi/patient_092_node_1.tif"
    coord_file = "/projects/bgmp/acrabtre/tumor-finder/output/coords_list_patient_092_node_1.csv"
    outfile = "./output/mini_WSI_patient_092_node_1_test.png"

    #wsi_file = "/projects/bgmp/acrabtre/tumor-finder/data/wsi/tcga/TCGA-GI-A2C8-01Z-00-DX1.09BD8AC9-645A-4C8B-9B36-77D833BDBA09.svs"
    #coord_file = "/projects/bgmp/acrabtre/tumor-finder/output/coords_list_TCGA-GI-A2C8-01Z-00-DX1.csv"
    #outfile = "./output/heatmap_WSI_TCGA-GI-A2C8-01Z-00-DX1_blues.png"
    save_heatmap(wsi_file, coord_file, outfile)