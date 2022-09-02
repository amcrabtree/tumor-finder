
import torch
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Predictor():
    """ Predicts image classification from a trained model.
    """
    def __init__(self, model):
        self.model = model
        self.stats_list = []

    def predict(self, data_loader)->np.ndarray:
        torch.set_grad_enabled(False) 
        pred_np = np.array([])
        for step, (images) in enumerate(data_loader):
            images = images.to(device)
            outputs = self.model(images)
            preds = outputs.argmax(-1).cpu().numpy()
            #probs = torch.nn.functional.softmax(outputs, dim=1)[:, 1].cpu()
            pred_np = np.concatenate((pred_np, preds), axis=0)
        return pred_np
