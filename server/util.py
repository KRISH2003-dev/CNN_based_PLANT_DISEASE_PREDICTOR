import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision import models
from PIL import Image
import pandas as pd
import torch.nn.functional as F

class PlantClassifier(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 512)
        self.relu1 = nn.ReLU(inplace=True)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.relu2 = nn.ReLU(inplace=True)
        self.drop2 = nn.Dropout(0.3)
        self.out = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.drop1(x)
        x = self.relu2(self.fc2(x))
        x = self.drop2(x)
        return self.out(x)

__model = None
__class_names = None
__leaf_model = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

def load_saved_artifacts():
    global __model, __class_names

    if __model is None:
        print("Loading PyTorch disease model...")

        df = pd.read_csv("labels.csv", encoding="utf-8-sig")
        __class_names = df["labels"].tolist()
        num_classes = len(__class_names)

        model = models.resnet50(weights="IMAGENET1K_V1")
        in_features = model.fc.in_features
        model.fc = PlantClassifier(in_features, num_classes)

        state_dict = torch.load("../model/best_resnet50_model.pth",
                                map_location=DEVICE)
        model.load_state_dict(state_dict)

        model.to(DEVICE)
        model.eval()

        __model = model
        print("Disease model loaded successfully.")

    return __model, __class_names

def load_leaf_model():
    global __leaf_model

    if __leaf_model is None:
        print("Loading leaf detector model...")

        model = models.resnet18(weights=None)

        in_features = model.fc.in_features

        model.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, 2)
        )

        state_dict = torch.load("../model/best_leaf_detector_model.pth",
                                map_location=DEVICE)

        model.load_state_dict(state_dict)

        model.to(DEVICE)
        model.eval()

        __leaf_model = model
        print("Leaf detector loaded.")

    return __leaf_model


def check_leaf_image(image_path):

    model = load_leaf_model()

    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(x)
        probs = F.softmax(outputs, dim=1)
        confidence, pred_idx = torch.max(probs, dim=1)

    classes = ["Leaf", "Not_Leaf"]

    return classes[pred_idx.item()], confidence.item()

def get_prediction(image_path):
    model, class_names = load_saved_artifacts()

    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(x)
        probs = F.softmax(outputs, dim=1)
        confidence, pred_idx = torch.max(probs, dim=1)

    predicted_label = class_names[pred_idx.item()]
    confidence_score = confidence.item()

    return predicted_label, confidence_score

if __name__ == "__main__":
    model, classes = load_saved_artifacts()
    print("Classes loaded:", classes[:5], "...")
