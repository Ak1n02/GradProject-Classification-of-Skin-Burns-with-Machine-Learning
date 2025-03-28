import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader


# Define the same model architecture as used during training
class BurnCNN(nn.Module):
    def __init__(self):
        super(BurnCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 28 * 28, 128)
        self.fc2 = nn.Linear(128, 3)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


# Load the trained model
model = BurnCNN()
model.load_state_dict(torch.load("burn_classification_cnn_gpu_pytorch.pth", map_location=torch.device("cpu")))
model.eval()

# Define class labels
class_labels = ["1st Degree Burn", "2nd Degree Burn", "3rd Degree Burn"]


# Define preprocessing function
def preprocess_image(img_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    img = Image.open(img_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)  # Add batch dimension
    return img_tensor


# Predict function
def predict(img_path):
    img_tensor = preprocess_image(img_path)
    with torch.no_grad():
        output = model(img_tensor)
    predicted_class = torch.argmax(output, dim=1).item()
    print(f"Predicted Burn Classification: {class_labels[predicted_class]}")


# Example usage
img_path = "test_akin_removed_onlyburn/second_degree/processed_2nddegree.png"
predict(img_path)

# Load test dataset for evaluation
test_dataset = datasets.ImageFolder(root="3131", transform=transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
]))
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# Evaluate the model on test data
def evaluate():
    correct = 0
    total = 0
    test_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_acc = 100.0 * correct / total
    test_loss /= len(test_loader)
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Loss: {test_loss:.4f}")

# Example usage
#evaluate()
