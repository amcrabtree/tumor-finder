from torchvision import transforms
import torch

def custom_tfm(model:str, data:str, input_size:int):
    """ Returns correct transform for model and data. 

    Args:
        model: name of model as specified in config file
        data: type of data the transformation is being applied to (either 'image' or 'label')
        input_size: pixel size of raw images (ex: 96 for 96x96 tiles)
    """
    T = None
    if data=='label':
        T = transforms.Lambda(lambda y: torch.tensor(y, dtype=torch.float))
    if data=='image':
        if model=='vgg16' or model=='NaturalSceneClassification':
            if input_size > 256:
                T = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ConvertImageDtype(torch.float), 
                    transforms.Normalize(
                        mean=[0.48235, 0.45882, 0.40784], 
                        std=[0.00392156862745098, 0.00392156862745098, 0.00392156862745098]),
                ])
            if input_size < 224:
                T = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Pad(224-input_size), # should make input image = 224x224
                    transforms.ConvertImageDtype(torch.float), 
                    transforms.Normalize(
                        mean=[0.48235, 0.45882, 0.40784], 
                        std=[0.00392156862745098, 0.00392156862745098, 0.00392156862745098]),
                        ])
            else: 
                T = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.CenterCrop(224),
                    transforms.ConvertImageDtype(torch.float), 
                    transforms.Normalize(
                        mean=[0.48235, 0.45882, 0.40784], 
                        std=[0.00392156862745098, 0.00392156862745098, 0.00392156862745098]),
                ])
    return T