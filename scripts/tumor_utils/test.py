import time
import torch
import torch.nn as nn

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Tester():
    """ Tests a model 
    """
    def __init__(self, model, config):
        self.config = config
        self.model = model
        self.stats_list = []
        self.test_loss = 0.0
        self.test_acc = 0.0
    
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
            if pred_is_accurate:
                true_count += 1
        accuracy = true_count / len(preds)
        return accuracy

    def test(self, test_loader):
        since = time.time() # track time
        with torch.no_grad():
            torch.set_grad_enabled(False)
            step_loss = []
            running_loss = 0.0
            running_accuracy = 0
            for step, (images, labels) in enumerate(test_loader):
                images, labels = [images.to(device), labels.to(device)]
                loss = self.loss_fn(images, labels)
                step_loss.append(loss.item())
                running_loss += loss.item() #* images.size(0)
                batch_acc = self.accuracy_fn(images, labels)
                running_accuracy += batch_acc

            ## print and save loss & acc
            self.test_loss = running_loss / len(test_loader)
            self.test_acc = running_accuracy / len(test_loader)
            self.stats_list.append(  # save loss info in stats_list attribute
                {'phase':'test',
                'test_loss':self.test_loss,
                'test_acc':self.test_acc
                })
            print(f'test Loss: {self.test_loss:.4f} Acc: {self.test_acc:.4f}')

        time_elapsed = time.time() - since
        print(f'Testing complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
