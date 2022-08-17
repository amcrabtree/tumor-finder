import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns; sns.set_theme()

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
    