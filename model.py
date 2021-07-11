import torch
import torch.nn as nn
from torchvision import models


class MRNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.pretrained_model = models.alexnet(pretrained=True)
        self.pooling_layer = nn.AdaptiveAvgPool2d(1)
        self.classifer = nn.Linear(256, 2)

    def forward(self, x):
        x = torch.squeeze(x, dim=0) 
        features = self.pretrained_model.features(x)
        pooled_features = self.pooling_layer(features)
        pooled_features = pooled_features.view(pooled_features.size(0), -1)
        flattened_features = torch.max(pooled_features, 0, keepdim=True)[0]
        output = self.classifer(flattened_features)
        return output

class MRNet2(nn.Module):
    def __init__(self):
        super().__init__()
        self.pretrained_model = models.alexnet(pretrained=True)
        self.pooling_layer = nn.AdaptiveAvgPool2d(1)
        self.drop =  nn.Dropout(p=0.2)
        self.classifer = nn.Linear(256, 2)

    def forward(self, x):
        x = torch.squeeze(x, dim=0)
        features = self.pretrained_model.features(x)
        pooled_features = self.pooling_layer(features)
        pooled_features = pooled_features.view(pooled_features.size(0), -1)
        flattened_features = torch.max(pooled_features, 0, keepdim=True)[0]
        flattened_features =  self.drop(flattened_features)

        output = self.classifer(flattened_features)
        return output

class MobileNetV3(nn.Module):    
    def __init__(self):
        super().__init__()
        mobilenet2 = models.mobilenet_v3_large(pretrained=True)
        modules = list(mobilenet2.children())[:-1]      # delete the last fc layer.
        self.mobilenet = nn.Sequential(*modules) 
        self.avgpool = nn.AvgPool2d((7,7))
        self.drop =  nn.Dropout(p=0.4)
        self.classifer = nn.Linear(960, 64)
        self.drop2 =  nn.Dropout(p=0.4)
        self.activation = nn.ReLU()
        self.classifer2 = nn.Linear(64, 2)

    def forward(self, x):
        features = self.mobilenet(x)
        #pooled_features = self.avgpool(features)
        features = self.drop(torch.squeeze(features) )
        #print(features.shape)
        output = self.classifer(features)
        output = self.drop2(output)
        output = self.activation(output)
        output = self.classifer2(output)
        return output
    
class MRNetSpecialMN(nn.Module):
    def __init__(self):
        super().__init__()
        mobilenet2 = models.mobilenet_v2(pretrained=True)
        modules = list(mobilenet2.children())[:-1]      # delete the last fc layer.
        self.mobilenet = nn.Sequential(*modules)
        self.pooling_layer = nn.AdaptiveAvgPool2d(1)
        self.drop1 =  nn.Dropout(p=0.2)
        self.classifer1 = nn.Linear(1280, 2)        
        #self.drop2 =  nn.Dropout(p=0.6)
        #self.activation = nn.ReLU()
        #self.classifer2 = nn.Linear(64, 5)

    def forward(self, x):
        x = torch.squeeze(x, dim=0)
        features = self.mobilenet(x)
        pooled_features = self.pooling_layer(features)
        pooled_features = pooled_features.view(pooled_features.size(0), -1)
        flattened_features = torch.max(pooled_features, 0, keepdim=True)[0]
        flattened_features =  self.drop1(flattened_features)
        output = self.classifer1(flattened_features)
        #output = self.drop2(output)
        #output = self.activation(output)
        #output = self.classifer2(output)
        return output

class ResNext(nn.Module):    
    def __init__(self):
        super().__init__()
        resnext50 = models.resnext50_32x4d(pretrained=True)
        modules = list(resnext50.children())[:-1]      
        self.resnext = nn.Sequential(*modules) 
        self.avgpool = nn.AvgPool2d((7,7))
        self.drop =  nn.Dropout(p=0.4)
        self.classifer = nn.Linear(2048, 256)
        self.drop2 =  nn.Dropout(p=0.4)
        self.activation = nn.ReLU()
        self.classifer2 = nn.Linear(256, 2)

    def forward(self, x):
        features = self.resnext(x)
        features = self.drop(torch.squeeze(features) )
        output = self.classifer(features)
        output = self.drop2(output)
        output = self.activation(output)
        output = self.classifer2(output)
        return output
    
