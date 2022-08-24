import time
import torch
import torch.nn as nn
import torchmetrics as TM
import pandas as pd
import numpy as np
from tumor_utils.viz import confusion_plot

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Tester():
    """ Tests a model 
    """
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.metrics = list(config['tester']['metrics'])
        self.criterion = getattr(torch.nn.modules.loss, config['loss'])()
        self.stats_list = []
        self.acc = 0.0
        self.f1 = 0.0
        self.precision = 0.0
        self.recall = 0.0
        self.confusion = np.zeros([2,2])

    def accuracy_fn(self, inputs, labels):
        """ Returns batch accuracy. 
        """
        outputs = self.model(inputs)
        _, preds = torch.max(outputs, dim=1) # best prediction (max prob.)
        true_count = torch.sum(torch.eq(preds, labels))
        accuracy = true_count / len(preds)
        return accuracy

    def accuracy_fn(self, inputs, labels):
        """ Returns batch accuracy. 
        """
        outputs = self.model(inputs)
        _, preds = torch.max(outputs, dim=1) # best prediction (max prob.)
        true_count = torch.sum(torch.eq(preds, labels))
        accuracy = true_count / len(preds)
        return accuracy

    def test(self, test_loader):
        torch.set_grad_enabled(False) 

        conf_metric = TM.ConfusionMatrix(num_classes=2)

        for step, (images, labels) in enumerate(test_loader):
            images, labels = [images.to(device), labels.to(device)]
            outputs = self.model(images)

            y_pred = outputs.argmax(-1).cpu()
            y_tgt = labels.long().cpu()
            confusion = conf_metric(y_pred, y_tgt)

        ## METRICS
        self.confusion = conf_metric.compute().cpu().numpy()
        print("Confusion matrix:\n", self.confusion)
        [tp,fp],[fn,tn] = self.confusion
        self.acc = (tp+tn)/(tp+tn+fp+fn)
        self.precision = tp / (tp+fp)
        self.recall = tp / (tp+fn)
        self.f1 = (2 * self.precision * self.recall) / (self.precision + self.recall)
        
        ## save metrics to stats dict
        self.stats_list.append( 
            {'model':self.model.__class__.__name__,
            'phase':'test',
            'tp':tp,
            'fp':fp,
            'fn':fn,
            'tn':tn,
            'acc':self.acc,
            'f1':self.f1,
            'precision':self.precision,
            'recall':self.recall
            })

        ## save stats info to file
        stats_df = pd.DataFrame.from_dict(self.stats_list)
        stats_df.to_csv(self.config['output']['metrics_file'], index=False)

        ## save confusion plot
        confusion_plot(
            matrix=self.confusion, 
            outfile=self.config['output']['confusion_plot'])
