import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import os
import re
import copy
from albumentations import (
    Compose, Resize, RandomResizedCrop, HorizontalFlip, Rotate, ColorJitter, CLAHE,ElasticTransform,
    GaussianBlur, RandomBrightnessContrast, HueSaturationValue, CoarseDropout, Normalize
)
from albumentations.pytorch import ToTensorV2
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, WeightedRandomSampler
from PIL import Image
from torchvision.transforms.v2.functional import horizontal_flip

import allmodels as modelss
class AlbumentationsDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.dataset = ImageFolder(root=root_dir)
        self.transform = transform

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
# --- Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset paths
train_data_dir = "enson_removed_preprocessed"  # Your full training dataset
test_data_dir = "test_yanik_yeni_removed_processed"  # Your completely separate test set
# Training parameters
batch_size = 32
max_epochs = 50

def prepare_data():
    # Define Albumentations transforms
    train_transform = Compose([
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
        ToTensorV2()
    ])

    # Use AlbumentationsDataset instead of ImageFolder
    train_dataset = AlbumentationsDataset(root_dir=train_data_dir, transform=train_transform)
    test_dataset = AlbumentationsDataset(root_dir=test_data_dir, transform=train_transform)

    print(f"\nDataset Sizes:")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    # Handle Class Imbalance
    train_labels = [label for _, label in train_dataset.dataset.samples]
    class_counts = np.bincount(train_labels)
    class_weights = 1. / class_counts
    weights = [class_weights[label] for label in train_labels]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, test_dataset

def train_model(train_loader, test_loader):
    model = modelss.BurnEfficientNet(num_classes=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    best_test_acc = 0.0
    best_model = None

    print("\nTraining on FULL Dataset...")
    for epoch in range(max_epochs):
        # Training phase
        model.train()
        train_loss, train_correct = 0.0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_correct += predicted.eq(labels).sum().item()

        train_acc = 100. * train_correct / len(train_loader.dataset)

        # Test evaluation
        model.eval()
        test_loss, test_correct = 0.0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                test_correct += predicted.eq(labels).sum().item()

        test_acc = 100. * test_correct / len(test_loader.dataset)
        scheduler.step(test_loss)

        print(f"Epoch {epoch + 1}/{max_epochs} | "
              f"Train Loss: {train_loss / len(train_loader):.4f} | "
              f"Train Acc: {train_acc:.2f}% | "
              f"Test Loss: {test_loss / len(test_loader):.4f} | "
              f"Test Acc: {test_acc:.2f}%")

        # Save best model
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_model = copy.deepcopy(model.state_dict())
            torch.save(best_model, "nisan26_burnCNN.pth")
            print(f"New best model saved with Test Acc: {test_acc:.2f}%")

    return model, best_test_acc


def evaluate_model(model, test_loader, test_dataset):
    model.eval()

    # Generate predictions
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Confusion Matrix and Classification Report
    cm = confusion_matrix(all_labels, all_preds)
    print("\nConfusion Matrix:")
    print(cm)

    report = classification_report(all_labels, all_preds, target_names=test_dataset.classes)
    print("\nClassification Report:")
    print(report)


def main():
    print("CUDA Available:", torch.cuda.is_available())

    # Prepare data
    train_loader, test_loader, test_dataset = prepare_data()

    # Train model
    model, best_test_acc = train_model(train_loader, test_loader)

    # Final evaluation
    print("\nTraining Complete!")
    print(f"Best Test Accuracy: {best_test_acc:.2f}%")

    # Load best model for final metrics
    model.load_state_dict(torch.load("modelB4_onthefly.pth"))

    # Evaluate
    evaluate_model(model, test_loader, test_dataset)

if __name__ == "__main__":
    main()