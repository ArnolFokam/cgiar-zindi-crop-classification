import timm
import torch
import torch.nn as nn


class Resnet50_V1(nn.Module):
    def __init__(self):
        super(Resnet50_V1, self).__init__()
        # Load a pretrained CNN (e.g., ResNet50)
        self.cnn = timm.create_model(
            'resnet50.a1_in1k',
            pretrained=True
        )
        
        # Modify the final classification layer 
        # to output a single regression value
        num_features = self.cnn.fc.in_features
        self.cnn.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Linear(512, 5),
        )

    def forward(self, x):
        x = self.cnn(x)
        return self.cnn(x)