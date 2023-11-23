import torch
import torch.nn as nn
import torchvision.models as models

class DenseNet_Custom(nn.Module):
    def __init__(self):
        super(DenseNet_Custom, self).__init__()
        # Load a DenseNet model
        self.cnn = models.densenet121(pretrained=False)  # Set pretrained=False if training from scratch

        # Modify the classifier layer
        num_features = self.cnn.classifier.in_features
        self.cnn.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.GELU(),
            nn.Linear(512, 5),  # Assuming you want 5 output classes
        )

    def forward(self, x):
        return self.cnn(x)