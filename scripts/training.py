#!/usr/bin/env python3 -u

"""
This script trains a machine learning model for tumor detection. 

    Author: Angela Crabtree
"""
import os
import sys
import pandas as pd
import json
from tumor_utils.data import TiledDataset,TryLoader # custom dataset class & test fxn
from tumor_utils.train import Trainer,salute # custom trainer class and print msg
from torch.utils.data import DataLoader
import torch
from torchvision.models import vgg16, VGG16_Weights
import torch.nn as nn
from torchvision import transforms

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#torch.backends.cudnn.benchmark = True # turn on autotuner for increase in overall speed

if __name__=='__main__':

    # initial run configurations 
    config_file = open(sys.argv[1], "r")
    config = json.loads(config_file.read())
    config_file.close()

    # create project directory if it doesn't exist
    project_dir = os.path.join(config['run_dir'], config['run_name'])
    if not os.path.exists(project_dir):
        os.makedirs(project_dir)

    # redirect stdout to logfile
    old_stdout = sys.stdout
    log_filename = os.path.join(project_dir, "training.log")
    log_file = open(log_filename,"w")
    sys.stdout = log_file

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
    training_set = TiledDataset(
        set_dir = os.path.join(config['tile_dir'], "train"),
        transform=img_transform,
        target_transform=label_transform
    )
    val_set = TiledDataset(
        set_dir = os.path.join(config['tile_dir'], "val"),
        transform=img_transform,
        target_transform=label_transform
    )

    # 2. load datasets into dataloaders
    train_loader = DataLoader(
        training_set, 
        batch_size=config['batch_size'], 
        num_workers=1, # 1 is increased speed compared to 0
        pin_memory=True, # must pin memory if num_workers>0
        shuffle=True, 
        drop_last=True)
    val_loader = DataLoader(
        val_set, 
        batch_size=config['batch_size'], 
        num_workers=1, 
        pin_memory=True,
        shuffle=True, 
        drop_last=True)
    #TryLoader(train_loader) # ensure dataset/dataloader are working correctly

    # 3. Load Model
    model = vgg16(weights=VGG16_Weights.DEFAULT)
    # modify dropout layers from proportion of zeroes =0.5 to =0.7
    model.classifier[2] = nn.Dropout(p=0.7, inplace=False)
    model.classifier[5] = nn.Dropout(p=0.7, inplace=False)
    # change tensor dimensions out of model
    num_ftrs = model.classifier[6].in_features # last layer's input size
    model.classifier[6] = nn.Linear(num_ftrs, len(val_set.label_dict), device=device) 
    model.to(device)
    print(f"\n\tMODEL:\n{'.' * 40} \n{model} \n{'.' * 40}\n")

    # 4. Train model
    salute()
    print("\nTraining model ...\n")
    trainer = Trainer(model, config)
    trainer.train(train_loader, val_loader)

    # 5. save the trained model to file
    model_scripted = torch.jit.script(model) # Export to TorchScript
    model_outfile = os.path.join(project_dir, "final_model.pt")
    model_scripted.save(model_outfile) # Save

    # 6. Save the training stats (loss and accuracy) to CSV
    stats_df = pd.DataFrame.from_dict(trainer.stats_list)
    stats_file = os.path.join(project_dir, "stats.csv")
    stats_df.to_csv(stats_file, index=False)
    print(f'\n\tSaved stats file to: \n\t\t{stats_file}\n')

    # 7. append additional info to config
    #config['n_train_tiles'] = len(training_set)
    #config['n_val_tiles'] = len(val_set)
    #config.update(model.config) # append model config info to config dict

    # close logfile and reset stdout
    sys.stdout = old_stdout
    log_file.close()