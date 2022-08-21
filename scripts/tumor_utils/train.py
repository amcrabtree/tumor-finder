import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 
from torch.optim import lr_scheduler
from torch.autograd import Variable

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Trainer():
    """ Trains a model 
    """
    def __init__(self, model, config):
        self.model = model
        self.config = config

        self.n_epochs = config['trainer']['epochs']
        self.criterion = eval(config['loss'])
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), 
            lr = config['optimizer']['args']['lr'], 
            momentum = config['optimizer']['args']['momentum'])
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, 
            step_size = config['lr_scheduler']['args']['step_size'], 
            gamma = config['lr_scheduler']['args']['gamma'])

        self.stats_list = []
        self.best_acc = 0.0
        self.current_epoch = 0
    
    def loss_fn(self, inputs, labels):
        outputs = self.model(inputs)
        if type(self.criterion) == torch.nn.modules.loss.CrossEntropyLoss:
            loss = self.criterion(outputs, labels.long()) # .long converts to int tensor
        else: # torch.nn.modules.loss.MSELoss
            _, preds = torch.max(outputs, dim=1) # best prediction (max prob.)
            preds = torch.tensor(preds, dtype=torch.float)
            loss = self.criterion(preds, labels) 
            loss = Variable(loss, requires_grad = True)
        return loss

    def accuracy_fn(self, inputs, labels):
        """ Returns batch accuracy. 
        """
        outputs = self.model(inputs)
        _, preds = torch.max(outputs, dim=1) # best prediction (max prob.)
        true_count = torch.sum(torch.eq(preds, labels))
        accuracy = true_count / len(preds)
        #print(f'accuracy: {true_count} / {len(preds)} = {accuracy}')
        return accuracy

    def train_step(self, train_loader):
        torch.set_grad_enabled(True) # track history if training
        running_loss = 0.0
        running_accuracy = 0
        for step, (images, labels) in enumerate(train_loader):
            self.optimizer.zero_grad() # zero the parameter gradients
            images, labels = [images.to(device), labels.to(device)]
            loss = self.loss_fn(images, labels)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            running_loss += loss.item() * images.size(0) 
            batch_acc = self.accuracy_fn(images, labels)
            running_accuracy += batch_acc.item()
    
        ## print and save epoch loss & acc
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = running_accuracy / len(train_loader)

        self.stats_list.append(  # save loss info in stats_list attribute
            {'epoch':self.current_epoch,
            'phase':'training',
            'epoch_loss':epoch_loss,
            'epoch_acc':epoch_acc
            })
        print(f'train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        #self.writer.add_scalar('train_loss', loss.item(),  self.current_epoch)
        return epoch_loss

    def val_step(self, val_loader):
        torch.set_grad_enabled(False)
        running_loss = 0.0
        running_accuracy = 0
        for step, (images, labels) in enumerate(val_loader):
            images, labels = [images.to(device), labels.to(device)]
            loss = self.loss_fn(images, labels)
            running_loss += loss.item() * images.size(0)
            batch_acc = self.accuracy_fn(images, labels)
            running_accuracy += batch_acc.item()

        ## print and save epoch loss & acc
        epoch_loss = running_loss / len(val_loader)
        epoch_acc = running_accuracy / len(val_loader)
        self.stats_list.append(  # save loss info in stats_list attribute
            {'epoch':self.current_epoch,
            'phase':'validation',
            'epoch_loss':epoch_loss,
            'epoch_acc':epoch_acc
            })
        print(f'val Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        #self.writer.add_scalar('val_loss', avg_loss,  self.current_epoch)
        return epoch_loss

    def train(self, train_loader, val_loader):
        best_val_loss = float('inf')
        for e in range(self.n_epochs):
            since = time.time() # track time/epoch
            print(f'\nEpoch {e}/{self.n_epochs - 1}')
            print('-' * 10)
            # train step
            self.train_step(train_loader)
            # eval step
            with torch.no_grad():
                val_loss = self.val_step(val_loader)
            # save model weights to file if validation loss is smallest
            if val_loss < best_val_loss:
                print(f"{val_loss} < {best_val_loss} saving model ...")
                torch.save(self.model, self.config['output']['final_model'])
                best_val_loss = val_loss
            self.current_epoch+=1
            time_elapsed = time.time() - since
            print(f'Epoch complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
            # save current stats  info to file
            stats_df = pd.DataFrame.from_dict(self.stats_list)
            stats_df.to_csv(self.config['output']['stats_file'], index=False)

        return best_val_loss


def salute():
    """ Encouragement for the user. """
    print("\n",
        "                 ___       ___     ___  __   __   __   ___     __   ___           ___              __\n",
        "|\/|  /\  \ /     |  |__| |__     |__  /  \ |__) /  ` |__     |__) |__     |  | |  |  |__|    \ / /  \ |  |\n" 
        "|  | /~~\  |      |  |  | |___    |    \__/ |  \ \__, |___    |__) |___    |/\| |  |  |  |     |  \__/ \__/.\n", 
        sep="")
