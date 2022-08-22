from torchvision.models import vgg16, VGG16_Weights
import torchvision
import torch.nn as nn
import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class vgg16_mod():
    """ VGG16 model, modified
    """
    def __init__(self):
        self.model = vgg16(weights=VGG16_Weights.DEFAULT)

        # modify dropout layers from proportion of zeroes =0.5 to =0.3
        #model.classifier[2] = nn.Dropout(p=0.3, inplace=False)
        #model.classifier[5] = nn.Dropout(p=0.3, inplace=False)

        # change tensor dimensions out of model
        num_ftrs = self.model.classifier[6].in_features # last layer's input size
        num_classes = 2
        self.model.classifier[6] = nn.Linear(num_ftrs, num_classes, device=device) 

class NaturalSceneClassification(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            
            nn.Conv2d(3, 32, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(32,64, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
        
            nn.Conv2d(64, 128, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.Conv2d(128 ,128, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            
            nn.Conv2d(128, 256, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.Conv2d(256,256, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            
            nn.Flatten(),
            nn.Linear(36864,1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512,2)
        )
    
    def forward(self, xb):
        return self.network(xb)