import timm
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import copy
import os
import re
import allmodels as modelss  # Assumes your model definitions (e.g., BurnResNet) are here

from albumentations import (
    Compose, Resize, RandomResizedCrop, HorizontalFlip, Rotate, ColorJitter, CLAHE,ElasticTransform,
    GaussianBlur, RandomBrightnessContrast, HueSaturationValue, CoarseDropout, Normalize
)
from albumentations.pytorch import ToTensorV2
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, WeightedRandomSampler
from PIL import Image
from torchvision.transforms.v2.functional import horizontal_flip

class AlbumentationsDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.dataset = ImageFolder(root=root_dir)
        self.transform = transform
        self.classes = self.dataset.classes  # Expose the classes attribute

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_path, label = self.dataset.samples[idx]
        image = Image.open(img_path).convert("RGB")
        image = np.array(image)

        if self.transform:
            augmented = self.transform(image=image)  # Pass as named argument
            image = augmented["image"]

        return image, label
# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("CUDA Available:", torch.cuda.is_available())

# Define dataset directory and data transforms
dataset_dir = "enson_removed_preprocessed"

basic_transform = Compose([
        RandomResizedCrop((224, 224), scale=(0.75, 1.0), ratio=(0.9, 1.1), p=1),
        HorizontalFlip(p=0.5),
        CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),
        Rotate(limit=20, p=0.7),
        ElasticTransform(alpha=120, sigma=120 * 0.1, p=0.5),
        ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.7),
        GaussianBlur(blur_limit=(3, 5), sigma_limit=(0.1, 2.0), p=0.4),
        HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.5),
        RandomBrightnessContrast(p=0.5),
        CoarseDropout(p=0.5),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        #Normalize(),
        ToTensorV2()
    ])
# Validation transforms with only resize and normalization
val_transform = Compose([
    Resize(224, 224),  # Simple resize without randomness
    Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    #Normalize(),
    ToTensorV2()
])

# Load the full training dataset
full_train_dataset = AlbumentationsDataset(root_dir=dataset_dir, transform=basic_transform)

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, reduction='mean'):
        """
        gamma: focusing parameter.
        weight: a manual rescaling weight given to each class.
        reduction: 'mean', 'sum', or 'none'
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction
        self.ce = nn.CrossEntropyLoss(weight=weight, reduction='none')

    def forward(self, inputs, targets):
        # Compute standard cross entropy loss without reduction
        ce_loss = self.ce(inputs, targets)
        # Convert to probabilities
        pt = torch.exp(-ce_loss)
        # Apply focal loss factor
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss



# Generate indices for training and validation splits
indices = list(range(len(full_train_dataset)))
train_indices, val_indices = train_test_split(indices, test_size=0.15, random_state=42)

# Create a separate validation dataset with validation transforms
val_dataset = AlbumentationsDataset(root_dir=dataset_dir, transform=val_transform)

# Create subsets
train_subset = Subset(full_train_dataset, train_indices)
val_subset = Subset(val_dataset, val_indices)  # Using val_dataset with val_transform

print(f"\nDataset sizes:")
print(f"Training samples: {len(train_indices)}")
print(f"Validation samples: {len(val_indices)}")

# Training function
def train(model, train_loader, criterion, optimizer):
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
    return running_loss / len(train_loader), 100.0 * correct / total

# Validation function
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
    return running_loss / len(val_loader), 100.0 * correct / total

# Early Stopping class
class EarlyStopping:
    def __init__(self, patience=5, delta=0.0):
        self.patience = patience
        self.delta = delta
        self.best_val_acc = 0.0
        self.epochs_no_improve = 0
        self.early_stop = False

    def __call__(self, val_acc):
        if val_acc > self.best_val_acc + self.delta:
            self.best_val_acc = val_acc
            self.epochs_no_improve = 0
            return False
        else:
            self.epochs_no_improve += 1
            if self.epochs_no_improve >= self.patience:
                self.early_stop = True
                return True
        return False


if __name__ == "__main__":
    torch.cuda.empty_cache()
    # Training parameters
    max_epochs = 80
    batch_size = 32

    # Implement cost-sensitive learning using focal loss.
    # Assuming class indices: 0: first_degree, 1: second_degree, 2: third_degree.
    # We'll give a higher cost for class 1.
    cost_weights = torch.tensor([1.0, 3.0, 3.0], dtype=torch.float32).to(device)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

    # Handle class imbalance with a WeightedRandomSampler
    train_labels = [train_subset.dataset.dataset.samples[i][1] for i in train_subset.indices]
    class_counts = np.bincount(train_labels)
    class_weights_sampler = 1.0 / class_counts.astype(np.float32)
    weights = [class_weights_sampler[label] for label in train_labels]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    # Create data loaders
    train_loader = DataLoader(train_subset, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

    # Initialize model and training components
    model = timm.create_model('regnety_080', pretrained=True)
    model.reset_classifier(num_classes=3)  # CRITICAL!!
    model = model.to(device)  # Move the model to the same device as the input data
    optimizer = optim.Adam(model.parameters(), weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    early_stopper = EarlyStopping(patience=40, delta=0.0)

    best_val_acc = 0.0
    best_model = None

    # Training loop
    for epoch in range(max_epochs):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer)
        val_loss, val_acc = validate(model, val_loader, criterion)
        scheduler.step(val_loss)

        print(f"Epoch {epoch + 1}/{max_epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        # Save the latest model after each epoch
        latest_model = copy.deepcopy(model.state_dict())
        torch.save(latest_model, f"regnet_epoch_{epoch + 1}.pth")
        print(f"Latest model saved at epoch {epoch + 1}")

        # Additionally, save a version that always gets overwritten to have the most recent
        torch.save(latest_model, "regnet_latest.pth")
        print(f"Latest model saved as regnet_latest.pth")

        # Evaluate the model after each epoch
        test_transform = Compose([
            Resize(224, 224),
            Normalize(),
            ToTensorV2()
        ])
        test_data_dir_a = "test_yanik_yeni_removed_processed"  # Your completely separate test set
        # Initialize the test dataset
        test_dataset = AlbumentationsDataset(root_dir=test_data_dir_a, transform=test_transform)

        # Initialize the DataLoader
        test_loader_a = DataLoader(test_dataset, batch_size=32, shuffle=False)

        from test_model import evaluate_model, plot_confusion_matrix

        cm, class_names = evaluate_model(model, test_loader_a, test_dataset)
        plot_confusion_matrix(cm, class_names)

        # Save best model
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_model = copy.deepcopy(model.state_dict())
            torch.save(best_model, "regnet_best.pth")
            print(f"New best model saved with Val Acc: {val_acc:.2f}%")

        if early_stopper(val_acc):
            print("Early stopping triggered")
            break

    # Load best model for final evaluation
    model.load_state_dict(torch.load("regnet_best.pth"))
    model.eval()

    # Generate predictions for evaluation
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Confusion Matrix and Classification Report
    cm = confusion_matrix(all_labels, all_preds)
    print("\nConfusion Matrix ephoch"+epoch+":")
    print(cm)

    report = classification_report(all_labels, all_preds, target_names=full_train_dataset.classes)
    print("\nClassification Report:")
    print(report)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=full_train_dataset.classes,
                yticklabels=full_train_dataset.classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

    print("\nTraining complete. Best validation accuracy: {:.2f}%".format(best_val_acc))