
import torch
import pandas as pd
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Predictor():
    """ Tests a model 
    """
    def __init__(self, model):
        self.model = model
        self.stats_list = []

    def predict(self, data_loader):
        torch.set_grad_enabled(False) 
        pred_list=[]
        for step, (images) in enumerate(data_loader):
            images = images.to(device)
            outputs = self.model(images)
            preds = outputs.argmax(-1).cpu().numpy()
            probs = torch.nn.functional.softmax(outputs, dim=1)[:, 1].cpu()
            pred_list.append(preds)
        pred_np = np.asarray(pred_list).flatten()
        return pred_np
