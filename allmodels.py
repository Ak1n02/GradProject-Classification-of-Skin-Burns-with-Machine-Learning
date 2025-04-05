import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import timm  # Library for pretrained models

class SimplifiedBurnCNN(nn.Module):
    def __init__(self):
        super(SimplifiedBurnCNN, self).__init__()
        # Convolutional Layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)

        # Global average pooling to reduce feature map to 1x1 per channel
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # A single fully connected layer for classification
        self.fc = nn.Linear(128, 3)

        # Activation and dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        # Global average pooling reduces the feature map to [batch, 128, 1, 1]
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # Flatten to [batch, 128]
        x = self.dropout(x)
        x = self.fc(x)  # Output logits for 3 classes
        return x

class EnhancedBurnCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Existing layers
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # Add attention module
        self.attention = nn.Sequential(
            nn.Conv2d(64, 32, 1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )

        # Modified later layers
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(128 * 28 * 28, 256)
        self.fc2 = nn.Linear(256, 3)

        # Add pooling layer
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))

        # Attention mechanism
        att_map = self.attention(x)
        x = x * att_map  # Focus on important regions

        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class BurnCNNEz(nn.Module):
    def __init__(self):
        super(BurnCNNEz, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 28 * 28, 128)
        self.fc2 = nn.Linear(128, 3)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

class BurnResNet(nn.Module):
    def __init__(self):
        super(BurnResNet, self).__init__()
        self.base = models.resnet18(pretrained=True)  # Load pretrained weights
        num_features = self.base.fc.in_features
        self.base.fc = nn.Linear(num_features, 3)  # Replace last layer

    def forward(self, x):
        return self.base(x)

class BurnCNN(nn.Module):
    def __init__(self):
        super(BurnCNN, self).__init__()
        # Convolutional Layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)  # Reduces spatial dims by 2 each time

        # Adjusted FC Layers (Intermediate Steps)
        self.fc1 = nn.Linear(128 * 32 * 32, 512)  # 128*32*32 = 131,072 → 512
        self.fc2 = nn.Linear(512, 256)  # 512 → 256
        self.fc3 = nn.Linear(256, 128)  # 256 → 128
        self.fc4 = nn.Linear(128, 3)  # 128 → 3 (output classes)

        # Regularization & Activation
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Convolutional Blocks
        x = self.pool(self.relu(self.bn1(self.conv1(x))))  # 256→128
        x = self.pool(self.relu(self.bn2(self.conv2(x))))  # 128→64
        x = self.pool(self.relu(self.bn3(self.conv3(x))))  # 64→32

        # Flatten & FC Layers
        x = x.view(x.size(0), -1)  # Flatten to [batch, 128*32*32]
        x = self.dropout(self.relu(self.fc1(x)))  # 131,072 → 512
        x = self.dropout(self.relu(self.fc2(x)))  # 512 → 256
        x = self.dropout(self.relu(self.fc3(x)))  # 256 → 128
        x = self.fc4(x)  # 128 → 3 (no activation/softmax yet)
        return x

class BurnResNet50(nn.Module):
    def __init__(self):
        super(BurnResNet50, self).__init__()
        self.base = models.resnet50(pretrained=True)  # Load pretrained weights
        num_features = self.base.fc.in_features
        self.base.fc = nn.Linear(num_features, 3)  # Replace last layer for 3 classes

    def forward(self, x):
        return self.base(x)


class BurnEfficientNet(nn.Module):
    def __init__(self, num_classes=3):
        super(BurnEfficientNet, self).__init__()
        self.model = timm.create_model("efficientnet_b4", pretrained=True)
        num_features = self.model.classifier.in_features  # Get input features of last layer
        self.model.classifier = nn.Linear(num_features, num_classes)  # Modify output layer

    def forward(self, x):
        return self.model(x)
