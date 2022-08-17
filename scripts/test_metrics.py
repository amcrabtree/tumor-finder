#!/usr/bin/env python3 -u

"""
This script runs test data through a machine learning model for tumor detection. 

    Author: Angela Crabtree
"""

######################## LIBRARIES ########################

import os
import sys
import pandas as pd
import json

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from tumor_utils.data import TiledDataset # custom dataset class 

from torch.utils.data import DataLoader
import torch
import torch.optim as optim 
from torchvision.models import vgg16, VGG16_Weights
import torch.nn as nn
from torchvision import transforms

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

######################## MAIN ########################
img_file = "/projects/bgmp/acrabtre/pytorch/dlwpt-code/data/p1ch2/bobby.jpg"
img_file = "/projects/bgmp/acrabtre/pytorch/amc/downloads/AdobeStock_473131207-1-scaled.jpeg"
# input image
img = Image.open(img_file)

# instantiate model (model instantiation must imply the model + weights + #of layers + #of units)
model = models.vgg16(weights=VGG16_Weights.DEFAULT)

#print(resnet)

# transform image
preprocess = T.Compose([
    T.Resize(256), 
    T.CenterCrop(224), 
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
        )
    ])
img_t = preprocess(img)
# reshape, crop, & normalize to create a mini-batch which the network expects
input_batch = torch.unsqueeze(img_t, 0) 

# move the input and model to GPU for speed if available
#if torch.cuda.is_available():
#    input_batch = input_batch.to('cuda')
#    model.to('cuda')

# perform forward pass using input image
model.eval() # puts model into inference mode
out = model(input_batch)

# retrieve class-label references
with open('/projects/bgmp/acrabtre/pytorch/dlwpt-code/data/p1ch2/imagenet_classes.txt') as f:
    labels = [line.strip() for line in f.readlines()]
_, index = torch.max(out, 1) # return index tensor of maximum score in output
percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100 # calculate confidence

# print final label and confidence percentage
print(f'Image contains a {labels[index[0]]} with {round(percentage[index[0]].item())}% confidence.')

######################## MAIN ########################

if __name__=='__main__':

    # initial run configurations 
    config_file = open(sys.argv[1], "r")
    config = json.loads(config_file.read())
    config_file.close()

    # 1. import datasets with custom Dataset class
    label_transform = transforms.Lambda(lambda y: torch.tensor(y, dtype=torch.float))
    img_transform = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float), 
        transforms.Normalize(
            mean=[0.48235, 0.45882, 0.40784], 
            std=[0.00392156862745098, 0.00392156862745098, 0.00392156862745098]),
    ])
    # define and transform datasets 
    test_set = TiledDataset(
        set_dir = os.path.join(config['tile_dir'], "test"),
        transform=img_transform,
        target_transform=label_transform
    )

    # 2. load datasets into dataloaders
    test_loader = DataLoader(
        test_set, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        drop_last=True)

    # 3. Load trained model from file
    model = torch.load(config['model_file'])
    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    # 4. Make predictions (perform forward pass using test images)
    model.eval() # puts model into inference mode
    out = model(input_batch)

    # 5. Assess performance
    #       scores

    #       confusion matrix




## make predictions
y_pred = model.predict(X_test).argmax(axis=1)
#print(Y_pred.argmax(axis=1))

## assess performance
#       scores
from sklearn import metrics
print("Accuracy score:", metrics.accuracy_score(y_test, y_pred))

#    confusion matrix
confusion_plot(y_pred, y_test)

