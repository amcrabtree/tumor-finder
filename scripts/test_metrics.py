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
from tumor_utils import data, tformer
import torch
from torch.utils.data import DataLoader

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if __name__=='__main__':

    # initial run configurations 
    config_file = open(sys.argv[1], "r")
    config = json.loads(config_file.read())
    config_file.close()

    # set full output filenames
    project_dir = os.path.join(config['run_dir'], config['run_name'])
    for el in config['output']:
        config['output'][el] = os.path.join(project_dir, config['output'][el])

    # 1. import datasets with custom Dataset class
    image_transform = tformer.custom_tfm(
            data='image', model=config['model']['arch'], input_size=256)
    label_transform = tformer.custom_tfm(
            data='label', model=config['model']['arch'], input_size=256)
    # define and transform datasets 
    test_set = getattr(data, config['data_loader']['type'])(
        set_dir = os.path.join(
            config['data_loader']['args']['data_dir'], 
            config['data_loader']['args']['test_subd']),
        transform = image_transform,
        target_transform = label_transform)

    # 2. load datasets into dataloaders
    test_loader = DataLoader(
        test_set, 
        batch_size = config['data_loader']['args']['batch_size'], 
        num_workers = 1, 
        pin_memory = True,
        shuffle = False, 
        drop_last = True)

    # 3. Load trained model from file
    model = torch.load(config['output']['running_model'])
    model.to(device)

    # save test summary to file
    old_stdout = sys.stdout # save original stdout
    test_log = open(config['output']['test_log'],"w")
    sys.stdout = test_log

    # 4. Make predictions (perform forward pass using test images)
    print("\nTesting model ...\n")
    model.eval() # puts model into inference mode
    tester = Tester(model, config)
    tester.test(test_loader)

    # close logfile and reset stdout
    sys.stdout = old_stdout
    test_log.close()
