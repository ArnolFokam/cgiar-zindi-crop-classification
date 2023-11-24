import torch
import torch.nn as nn
import torchvision.models as models

class DenseNet_Custom(nn.Module):
    def __init__(self):
        super(DenseNet_Custom, self).__init__()
        # Load a DenseNet model
        self.cnn = models.densenet201(pretrained=True)  # Set pretrained=False if training from scratch - changed to 201

        # Modify the classifier layer
        num_features = self.cnn.classifier.in_features
        self.cnn.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.GELU(),
            nn.Dropout(0.2),  # Add dropout here, 20% probability
            nn.Linear(512, 256),  # Additional layer
            nn.GELU(),
            nn.Dropout(0.3),  # Add dropout here, 30% probability
            nn.Linear(256, 5),  # Assuming you want 5 output classes
        )

    def forward(self, x):
        return self.cnn(x)