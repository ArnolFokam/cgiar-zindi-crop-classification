import timm
import torch.nn as nn

class EfficientNetB4_Custom(nn.Module):
    def __init__(self):
        super(EfficientNetB4_Custom, self).__init__()
        # Load a pre-trained EfficientNet-B4 model
        self.cnn = timm.create_model('tf_efficientnet_b4_ns', pretrained=True)

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
