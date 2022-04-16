import torch
import torch.nn as nn
import torchvision.models as torch_models

def ResNet18(x_dim, y_dim):
    policy = torch_models.resnet18()
    policy.conv1 = nn.Conv2d(10, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    policy.fc = nn.Linear(in_features=512, out_features=x_dim*y_dim, bias=True) 
    return policy  

