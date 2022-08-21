import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns; sns.set_theme()
from torch.utils.data import Dataset
import random
import torch
import pandas as pd
from plotnine import *

def ViewNpyImg(npy_file:str):
    """ Display .npy tile image as pop-up. """
    tile_np = np.load(npy_file)
    plt.imshow(tile_np, interpolation='nearest')
    plt.show()

def confusion_plot(Y_labels: np.ndarray, Y_pred: np.ndarray):
    '''Plot confusion matrix for multiclass classification.'''
    matrix = confusion_matrix(Y_labels, Y_pred)
    # Plot non-normalized confusion matrix
    sns.heatmap(matrix, annot=True, fmt="d")
    plt.title('Confusion Matrix')
    plt.xlabel('Prediction')
    plt.ylabel('Ground Truth')
    plt.show()
    print('Number of Incorrect Guesses:',np.sum(Y_labels!=Y_pred))
    print('Number of Total Predictions:', np.shape(Y_labels)[0])

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
