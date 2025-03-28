import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import KFold
import numpy as np

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("CUDA Available:", torch.cuda.is_available())
print("CUDA Device Count:", torch.cuda.device_count())
print("CUDA Device Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU detected")
print("Current Device:", torch.cuda.current_device() if torch.cuda.is_available() else "CPU only")

# Define transformations for the dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load dataset
dataset = datasets.ImageFolder(root="final_data_set_no_bg_augmented_onlyburnt_whitebg", transform=transform)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import KFold

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
        x = x.view(x.size(0), -1)  # Flatten
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

def train(model, train_loader, criterion, optimizer, scheduler):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_acc = 100.0 * correct / total

    return train_loss, train_acc


def validate(model, val_loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    val_loss = running_loss / len(val_loader)
    val_acc = 100.0 * correct / total

    return val_loss, val_acc

def cross_validate_model(k, epochs=20, batch_size=64):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    indices = np.arange(len(dataset))

    fold_accuracies = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(indices), 1):
        print(f"\nTraining on Fold {fold}/{k}...")

        # Define data samplers
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)

        # Create data loaders
        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
        val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)

        # Initialize model, loss function, optimizer, and learning rate scheduler
        model = BurnCNN().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), weight_decay=0.01)  # L2 Regularization
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

        # Early stopping
        early_stopping = EarlyStopping(patience=5, delta=0.01)

        best_val_acc = 0.0
        for epoch in range(epochs):
            train_loss, train_acc = train(model, train_loader, criterion, optimizer, scheduler)
            val_loss, val_acc = validate(model, val_loader, criterion)

            print(f"Epoch {epoch + 1}/{epochs} - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

            # Check early stopping criteria
            stop_training = early_stopping(val_acc, model)
            if stop_training:
                print(f"Early stopping at epoch {epoch + 1}")
                early_stopping.restore_best_weights(model)
                break

            best_val_acc = max(best_val_acc, val_acc)

        fold_accuracies.append(best_val_acc)
        print(f"Fold {fold} Best Accuracy: {best_val_acc:.4f}")

    # Compute final cross-validation accuracy
    mean_accuracy = np.mean(fold_accuracies)
    std_dev = np.std(fold_accuracies)

    print("\nFinal Cross-Validation Accuracy:")
    print(f"Mean Accuracy: {mean_accuracy:.4f}")
    print(f"Standard Deviation: {std_dev:.4f}")

    # Save final model
    torch.save(model.state_dict(), "burn_classification_cnn_gpu_pytorch.pth")


class EarlyStopping:
    def __init__(self, patience, delta=0):
        """
        Args:
            patience (int): Number of epochs to wait for improvement.
            delta (float): Minimum change to qualify as an improvement.
        """
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.epochs_no_improve = 0
        self.best_model_wts = None

    def __call__(self, val_acc, model):
        # If this is the first time, we consider this as the best score
        if self.best_score is None:
            self.best_score = val_acc
            self.best_model_wts = model.state_dict()
            return False

        # Check if the current validation accuracy is better than the best score
        if val_acc > self.best_score + self.delta:
            self.best_score = val_acc
            self.best_model_wts = model.state_dict()
            self.epochs_no_improve = 0
            return False
        else:
            self.epochs_no_improve += 1

        # If no improvement for 'patience' number of epochs, stop training
        if self.epochs_no_improve >= self.patience:
            return True

        return False

    def restore_best_weights(self, model):
        model.load_state_dict(self.best_model_wts)


# Run training
if __name__ == "__main__":
    cross_validate_model(k=10, epochs=20)