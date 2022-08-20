#!/usr/bin/env python3 -u

"""
This script trains a machine learning model for tumor detection. 

    Author: Angela Crabtree
"""
import os
import sys
import json
import shutil
from tumor_utils.data import TiledDataset,try_loading # custom dataset class & test fxn
from tumor_utils.train import Trainer,salute # custom trainer class and print msg
import tumor_utils # custom dataset for 96x96 Pcam tiles
from torch.utils.data import DataLoader
import torch
from torchvision.models import vgg16, VGG16_Weights
import torch.nn as nn
from torchvision import transforms
from torchsummary import summary

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#torch.backends.cudnn.benchmark = True # turn on autotuner for increase in overall speed

if __name__=='__main__':

    # initial run configurations 
    config_file = open(sys.argv[1], "r")
    config = json.loads(config_file.read())
    config_file.close()
    
    # set full output filenames
    project_dir = os.path.join(config['run_dir'], config['run_name'])
    for el in config['output']:
        config['output'][el] = os.path.join(project_dir, config['output'][el])

    # create project directory if it doesn't exist
    if not os.path.exists(project_dir):
        os.makedirs(project_dir)
        
    # make copy of config file (will add extra info to it if epochs complete)
    shutil.copyfile(sys.argv[1], config['output']['config_outfile'])

    # 1. import datasets with custom Dataset class
    #MyDataset = eval(config['data_loader']['dataset_type'])
    training_set = tumor_utils.pcamv1.PCam(
        set_dir = os.path.join(
            config['data_loader']['args']['data_dir'], 
            config['data_loader']['args']['train_subd'])
    )
    val_set = tumor_utils.pcamv1.PCam(
        set_dir = os.path.join(
            config['data_loader']['args']['data_dir'], 
            config['data_loader']['args']['val_subd'])
    )

    # 2. load datasets into dataloaders
    train_loader = DataLoader(
        training_set, 
        batch_size = config['data_loader']['args']['batch_size'], 
        num_workers = 1, # 1 is increased speed compared to 0
        pin_memory = True, # must pin memory if num_workers>0
        shuffle = config['data_loader']['args']['shuffle'], 
        drop_last = True)
    val_loader = DataLoader(
        val_set, 
        batch_size = config['data_loader']['args']['batch_size'], 
        num_workers = 1, 
        pin_memory = True,
        shuffle = config['data_loader']['args']['shuffle'], 
        drop_last = True)
    #try_loading(train_loader) # ensure dataset/dataloader are working correctly

    # 3. Load Model
    model = vgg16(weights=VGG16_Weights.DEFAULT)
    # modify dropout layers from proportion of zeroes =0.5 to =0.7
    model.classifier[2] = nn.Dropout(p=0.7, inplace=False)
    model.classifier[5] = nn.Dropout(p=0.7, inplace=False)
    # change tensor dimensions out of model
    num_ftrs = model.classifier[6].in_features # last layer's input size
    num_classes = len(val_set.label_dict)
    #if num_classes == 2: num_classes=1 # if only 2 classes, make it binary
    model.classifier[6] = nn.Linear(num_ftrs, num_classes, device=device) 
    model.to(device)

    # save model summary to file
    img, _ = next(iter(train_loader))
    b,c,h,w = img.size()
    INPUT_SIZE = (c,h,w)
    old_stdout = sys.stdout # save original stdout
    model_summary = open(config['output']['model_summary_file'],"w")
    sys.stdout = model_summary
    print(model, "\n\n\n")
    summary(model, INPUT_SIZE)
    model_summary.close()
    
    # 4. Train model
    train_log = open(config['output']['train_logfile'],"w")
    sys.stdout = train_log
    salute()
    print("\nTraining model ...\n")
    trainer = Trainer(model, config)
    trainer.train(train_loader, val_loader)

    # 5. save the trained model to file
    model_scripted = torch.jit.script(model) # Export to TorchScript
    model_scripted.save(config['output']['final_model']) # Save

    # 6. append additional info to config and save to output config in project_dir
    config['notes']['device'] = str(device)
    config['notes']['n_train_tiles'] = len(training_set)
    config['notes']['n_val_tiles'] = len(val_set)
    json_object = json.dumps(config, indent=4) # Serializing json
    #config.update(model.config) # append model config info to config dict
    with open(config['output']['config_outfile'], "w") as f:
        f.write(json_object)
    
    # close logfile and reset stdout
    sys.stdout = old_stdout
    train_log.close()