import torch
import torch.nn as nn
import torchvision.models as models

class PlantDiseaseModel(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(PlantDiseaseModel, self).__init__()
        # Using ResNet50 as base model
        self.base_model = models.resnet50(pretrained=pretrained)
        
        # Replace final layer
        num_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.base_model(x)

# For mobile deployment
class LightweightModel(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(LightweightModel, self).__init__()
        # Using MobileNetV2 for mobile deployment
        self.base_model = models.mobilenet_v2(pretrained=pretrained)
        
        # Replace classifier
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.base_model.last_channel, num_classes)
        )
    
    def forward(self, x):
        return self.base_model(x)