import torch
import torch.nn as nn
import os
import torchvision.models as models
import torch.nn.functional as F
from torch.autograd import Variable



class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        
        model = models.vgg16(pretrained = True)
        first = model.features[0]
        features  = model.features[1:31]

        w1 = first.state_dict()['weight'][:,0,:,:]
        w2 = first.state_dict()['weight'][:,1,:,:]
        w3 = first.state_dict()['weight'][:,2,:,:]
        w4 = w1 + w2 + w3
        w4 = w4.unsqueeze(1)

        first_conv  = nn.Conv2d(1, 64 , 3, padding = (1,1))
        first_conv.weight = torch.nn.Parameter(w4,requires_grad = True)
        first_conv.bias = torch.nn.Parameter(first.state_dict()['bias'],requires_grad = True)
        
        self.first_convlayer = first_conv
        self.features = nn.Sequential(features)
        self.classifier = nn.Linear(512, 7)

    def forward(self, x):
        out = self.first_convlayer(x)
        out = self.features(out)
        out = out.view(-1,512)
        out = F.dropout(out, p=0.5, training=self.training)
        out = self.classifier(out)
        return out
