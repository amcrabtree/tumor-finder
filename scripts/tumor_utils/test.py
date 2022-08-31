
import torch
import torchmetrics as TM
import pandas as pd
import numpy as np
from tumor_utils.viz import confusion_plot, roc_plot

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Tester():
    """ Tests a model 
    """
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.criterion = getattr(torch.nn.modules.loss, config['loss'])()
        self.stats_list = []
        self.acc = 0.0
        self.f1 = 0.0
        self.precision = 0.0
        self.recall = 0.0
        self.confusion = np.zeros([2,2])

    def test(self, test_loader):
        torch.set_grad_enabled(False) 

        conf_metric = TM.ConfusionMatrix(num_classes=2)
        roc = TM.ROC(pos_label=1)
        auroc = TM.AUROC(pos_label=1)

        for step, (images, labels) in enumerate(test_loader):
            images, labels = [images.to(device), labels.to(device)]
            outputs = self.model(images)
            preds = outputs.argmax(-1).cpu()
            probs = torch.nn.functional.softmax(outputs, dim=1)[:, 1].cpu()
            target = labels.long().cpu()
            confusion = conf_metric(preds, target)
            fpr, tpr, thresholds = roc(probs, target)
            auroc_score = auroc(probs, target)

        ## METRICS
        self.confusion = conf_metric.compute().cpu().numpy()
        self.fpr, self.tpr, self.thresholds = roc.compute()
        self.fpr = self.fpr.cpu().numpy()
        self.tpr = self.tpr.cpu().numpy()
        self.thresholds = self.thresholds.cpu().numpy()
        self.auroc_score = auroc.compute().cpu().numpy()
        [tn,fp],[fn,tp] = self.confusion
        self.acc = (tp+tn)/(tp+tn+fp+fn)
        self.precision = tp / (tp+fp)
        self.recall = tp / (tp+fn)
        self.f1 = (2 * self.precision * self.recall) / (self.precision + self.recall)

        print("Confusion matrix:\n", self.confusion)
        
        ## save metrics to stats dict
        self.stats_list.append( 
            {'model':self.model.__class__.__name__,
            'tp':tp,
            'fp':fp,
            'fn':fn,
            'tn':tn,
            'acc':self.acc.round(4),
            'f1':self.f1.round(4),
            'precision':self.precision.round(4),
            'recall':self.recall.round(4), 
            'auroc':self.auroc_score.round(4)
            })

        ## save stats info to files
        stats_df = pd.DataFrame.from_dict(self.stats_list)
        stats_df.to_csv(self.config['output']['metrics_file'], index=False)

        roc_df = pd.DataFrame ({
            'model':self.model.__class__.__name__,
            'fpr':self.fpr,
            'tpr':self.tpr,
            'thresholds':self.thresholds})
        roc_df.to_csv(self.config['output']['roc_file'], index=False)

        ## save plots
        confusion_plot(
            matrix=self.confusion, 
            outfile=self.config['output']['confusion_plot'])
        roc_plot(
            roc_df, 
            auroc=self.auroc_score.round(3), 
            outfile=self.config['output']['roc_plot'])
