from torchvision import models
import torchvision
import torch.nn as nn
import torch

""" This is where all the model architectures are stored, as the config file refers to. 

Script adapted from: https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
"""

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def initialize_model(model_name:str, num_classes:int=2, feature_extract:bool=False, 
use_pretrained:bool=True):
    """ Loads a model. 

    Args:
        model_name: must be one of the following - 
            vgg16, resnet, alexnet, vgg, squeezenet, densenet, inception, NaturalSceneClassification
        num_classes: number of output classes
        feature_extract: Flag for feature extracting. When False, we finetune the whole model,
            when True we only update the reshaped layer params
        use_pretrained: use pre-trained weights instead of random weights
    """
    model_ft = None
    input_size = 0
    weights = None # random weights

    if model_name == "vgg16":
        """ VGG-16 with pre-trained weights
        """
        if use_pretrained == True: weights = models.VGG16_Weights.DEFAULT
        model_ft = models.vgg16(weights=weights)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "resnet":
        """ Resnet18
        """
        if use_pretrained == True: weights = models.ResNet18_Weights.IMAGENET1K_V1
        model_ft = models.resnet18(weights=weights)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        if use_pretrained == True: weights = models.AlexNet_Weights.IMAGENET1K_V1
        model_ft = models.alexnet(weights=weights)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        if use_pretrained == True: weights = models.VGG11_BN_Weights.IMAGENET1K_V1
        model_ft = models.vgg11_bn(weights=weights)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        if use_pretrained == True: weights = models.SqueezeNet1_0_Weights.IMAGENET1K_V1
        model_ft = models.squeezenet1_0(weights=weights)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        if use_pretrained == True: weights = models.DenseNet121_Weights.IMAGENET1K_V1
        model_ft = models.densenet121(weights=weights)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        if use_pretrained == True: weights = models.Inception_V3_Weights.IMAGENET1K_V1
        model_ft = models.inception_v3(weights=weights)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    elif model_name == "NaturalSceneClassification":
        """ Natural Scene Classification 
        """
        input_size = 224
        model_ft = NaturalSceneClassification()

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size




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
            nn.Linear(200704,1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512,2)
        )
    
    def forward(self, xb):
        return self.network(xb)
