import time
import torch
import torch.nn as nn
import torchmetrics as TM
import pandas as pd

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
        self.test_acc = 0.0

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
        since = time.time() # track time
        with torch.no_grad():
            torch.set_grad_enabled(False)
            accuracy = 0.0
            f1_score = 0.0
            auroc = 0

            for step, (images, labels) in enumerate(test_loader):
                images, labels = [images.to(device), labels.to(device)]
                outputs = self.model(images)

                y_pred = outputs.argmax(-1).cpu()
                #y_pred = outputs
                y_tgt = labels.long().cpu()

                accuracy = TM.Accuracy()(y_pred, y_tgt)
                f1_score = TM.F1Score()(y_pred, y_tgt)
                prere = TM.functional.precision_recall(y_pred, y_tgt)
                print(f'Batch: {step} Acc: {accuracy:.4f} f1_score: {f1_score:.4f} auroc: {prere}')

            time_elapsed = time.time() - since
            print(f'Testing complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

"""
            ## METRICS
            self.test_acc = running_accuracy / len(test_loader)
            self.precision = FM.precision_recall(images, labels, average='micro')

            ## save metrics to stats dict
            self.stats_list.append(  # save loss info in stats_list attribute
                {'model':self.model.__class__.__name__,
                'phase':'test',
                'acc':self.test_acc,
                })
            print(f'test Acc: {self.test_acc:.4f}')

            ## save stats info to file
            stats_df = pd.DataFrame.from_dict(self.stats_list)
            stats_df.to_csv(self.config['output']['metrics_file'], index=False)
"""