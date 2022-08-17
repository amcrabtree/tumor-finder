#!/usr/bin/env python3 -u

"""
This script runs test data through a machine learning model for tumor detection. 

    Author: Angela Crabtree
"""

import os
import sys
import json
import pandas as pd
from tumor_utils.data import TiledDataset # custom dataset class 
from tumor_utils.test import Tester # custom class for testing
from torch.utils.data import DataLoader
import torch
from torchvision import transforms
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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
    model.to(device)

    # 4. Make predictions (perform forward pass using test images)
    print("\nTesting model ...\n")
    model.eval() # puts model into inference mode
    trainer = Tester(model, config)
    trainer.test(test_loader)

    # 5. Save the training stats (loss and accuracy) to CSV
    stats_df = pd.DataFrame.from_dict(trainer.stats_list)
    project_dir = os.path.join(config['out_dir'], config['run_name'])
    stats_file = os.path.join(project_dir, "test_stats.csv")
    stats_df.to_csv(stats_file, index=False)
    print(f'\n\tSaved stats file to: \n\t\t{stats_file}\n')

    # 6. Generate Plots
    ## confusion matrix
    #confusion_plot(y_pred, y_test)
