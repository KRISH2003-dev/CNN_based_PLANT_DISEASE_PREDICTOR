import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_DIR = "../dataset/Binary_Plant_Non-Plant_Dataset"
MODEL_SAVE_PATH = "../best_leaf_detector_model.pth"
LABELS_SAVE_PATH = "../server/leaf_labels.csv"

IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 25
FREEZE_EPOCHS = 5

LR_HEAD = 1e-3
LR_BACKBONE = 1e-5
WEIGHT_DECAY = 1e-4
PATIENCE = 6

def main():

    os.makedirs("model", exist_ok=True)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(25),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
        transforms.ColorJitter(0.3, 0.3, 0.3),
        transforms.RandomGrayscale(p=0.1),
        transforms.GaussianBlur(3),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    train_dataset = datasets.ImageFolder(
        os.path.join(DATA_DIR, "train"),
        transform=train_transform
    )

    val_dataset = datasets.ImageFolder(
        os.path.join(DATA_DIR, "val"),
        transform=val_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    class_names = train_dataset.classes
    num_classes = len(class_names)

    pd.DataFrame({"labels": class_names}).to_csv(LABELS_SAVE_PATH, index=False)

    print("Classes:", class_names)

    targets = train_dataset.targets

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(targets),
        y=targets
    )

    class_weights_tensor = torch.tensor(class_weights).float().to(DEVICE)

    model = models.efficientnet_b0(weights="IMAGENET1K_V1")

    for param in model.features.parameters():
        param.requires_grad = False

    in_features = model.classifier[1].in_features

    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes)
    )

    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights_tensor,
        label_smoothing=0.1
    )

    optimizer = optim.AdamW(
        model.classifier.parameters(),
        lr=LR_HEAD,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS
    )

    scaler = GradScaler()

    best_acc = 0
    best_model_wts = copy.deepcopy(model.state_dict())
    early_stop_counter = 0

    for epoch in range(EPOCHS):

        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print("-" * 40)

        if epoch == FREEZE_EPOCHS:

            print("Unfreezing backbone...")

            for param in model.features.parameters():
                param.requires_grad = True

            optimizer = optim.AdamW([
                {"params": model.features.parameters(), "lr": LR_BACKBONE},
                {"params": model.classifier.parameters(), "lr": LR_HEAD}
            ], weight_decay=WEIGHT_DECAY)

        model.train()

        running_loss = 0
        running_correct = 0

        for inputs, labels in train_loader:

            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            with autocast(device_type="cuda"):

                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()

            preds = torch.argmax(outputs, dim=1)

            running_loss += loss.item() * inputs.size(0)
            running_correct += torch.sum(preds == labels)

        train_loss = running_loss / len(train_dataset)
        train_acc = running_correct.double() / len(train_dataset)

        model.eval()

        val_correct = 0
        val_preds = []
        val_labels = []

        with torch.no_grad():

            for inputs, labels in val_loader:

                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(inputs)

                preds = torch.argmax(outputs, dim=1)

                val_correct += torch.sum(preds == labels)

                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_acc = val_correct.double() / len(val_dataset)

        scheduler.step()

        print(f"Train Loss : {train_loss:.4f}")
        print(f"Train Acc  : {train_acc:.4f}")
        print(f"Val Acc    : {val_acc:.4f}")

        if val_acc > best_acc:

            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())

            torch.save(best_model_wts, MODEL_SAVE_PATH)

            print("Best model saved!")

            early_stop_counter = 0

        else:

            early_stop_counter += 1

        if early_stop_counter >= PATIENCE:

            print("Early stopping triggered.")
            break

    model.load_state_dict(best_model_wts)

    print("\nTraining Complete")
    print("Best Validation Accuracy:", best_acc.item())

    print("\nClassification Report")
    print(classification_report(val_labels, val_preds, target_names=class_names))

    print("\nConfusion Matrix")
    print(confusion_matrix(val_labels, val_preds))


if __name__ == "__main__":
    main()