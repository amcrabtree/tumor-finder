import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim 
import torch.optim as optim 
from torch.optim import lr_scheduler

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Trainer():
    """ Trains a model 
    """
    def __init__(self, model, config):
        self.config = config
        self.model = model
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.0001, momentum=0.9)
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=7, gamma=0.1)
        self.n_epochs = config['n_epochs']
        self.stats_list = []
        self.best_acc = 0.0
        self.current_epoch = 0
    
    def loss_fn(self, inputs, labels):
        criterion = nn.CrossEntropyLoss()
        outputs = self.model(inputs)
        loss = criterion(outputs, labels.long()) # .long converts to scalar
        return loss

    def accuracy_fn(self, inputs, labels):
        """ Returns batch accuracy. 
        """
        outputs = self.model(inputs)
        _, preds = torch.max(outputs, dim=1) # best prediction (max prob.)
        true_count = 0
        for i in range(len(preds)):
            pred_is_accurate = (preds[i] == labels.data[i]).data
            #print(f'{preds[i]} == {labels.data[i]}: correct? {preds[i] == labels.data[i]}')
            if pred_is_accurate:
                true_count += 1
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
            running_loss += loss.item() #* images.size(0)
            self.optimizer.step()
            self.scheduler.step()
            batch_acc = self.accuracy_fn(images, labels)
            running_accuracy += batch_acc
            # if (step+1)% self.grad_acum_steps == 0:
            #     self.optimizer.step()
            #     self.optimizer.zero_grad()
    
        ## print and save epoch loss & acc
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = running_accuracy / len(train_loader)
        self.stats_list.append(  # save loss info in stats_list attribute
            {'epoch':self.current_epoch,
            'phase':'training',
            'epoch_loss':epoch_loss,
            'epoch_acc':epoch_acc
            })
        print('train_loss.item', loss.item())
        print('train epoch_loss', epoch_loss)
        print('train epoch_acc', epoch_acc)
        #self.writer.add_scalar('train_loss', loss.item(),  self.current_epoch)
        return loss.item()

    def val_step(self, val_loader):
        torch.set_grad_enabled(False)
        step_loss = []
        running_loss = 0.0
        running_accuracy = 0
        for step, (images, labels) in enumerate(val_loader):
            images, labels = [images.to(device), labels.to(device)]
            loss = self.loss_fn(images, labels)
            step_loss.append(loss.item())
            running_loss += loss.item() #* images.size(0)
            batch_acc = self.accuracy_fn(images, labels)
            running_accuracy += batch_acc
        avg_loss = np.mean(step_loss)

        ## print and save epoch loss & acc
        epoch_loss = running_loss / len(val_loader)
        epoch_acc = running_accuracy / len(val_loader)
        self.stats_list.append(  # save loss info in stats_list attribute
            {'epoch':self.current_epoch,
            'phase':'validation',
            'epoch_loss':epoch_loss,
            'epoch_acc':epoch_acc
            })
        print('val_loss, avg', avg_loss)
        print('val epoch_loss', epoch_loss)
        print('val epoch_acc', epoch_acc)
        #self.writer.add_scalar('val_loss', avg_loss,  self.current_epoch)
        return avg_loss

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
                project_dir = os.path.join(
                    self.config['out_dir'], self.config['run_name'])
                model_path = os.path.join(
                    project_dir, "running_model.pt")
                torch.save(self.model, model_path)
                best_val_loss = val_loss
            self.current_epoch+=1
            time_elapsed = time.time() - since
            print(f'Epoch complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

        return best_val_loss
