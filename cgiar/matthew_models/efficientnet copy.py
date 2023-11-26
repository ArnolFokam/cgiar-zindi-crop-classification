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
            nn.Dropout(0.5),  # Add dropout here, 20% probability -> upped to 50%
            nn.Linear(512, 512),  # Additional layer
            nn.GELU(),
            nn.Dropout(0.5),  # Add dropout here, 30% probability -> upped to 50%
            nn.Linear(512, 5),  # Assuming you want 5 output classes
        )

    def forward(self, x):
        return self.cnn(x)
