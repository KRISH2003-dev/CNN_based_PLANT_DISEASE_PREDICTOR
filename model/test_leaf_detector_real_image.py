import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import torch.nn.functional as F
import os

MODEL_PATH = "../best_leaf_detector_model.pth"
LABELS_PATH = "../server/leaf_labels.csv"

IMAGE_PATH = r"C:\Users\Accer\Downloads\my_signature.jpeg"

THRESHOLD = 0.80
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


labels_df = pd.read_csv(LABELS_PATH)
class_names = labels_df["labels"].tolist()

num_classes = len(class_names)


model = models.efficientnet_b0(weights=None)

in_features = model.classifier[1].in_features

model.classifier = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(in_features, 256),
    nn.ReLU(inplace=True),
    nn.Dropout(0.4),
    nn.Linear(256, num_classes)
)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


if not os.path.exists(IMAGE_PATH):
    print("Image path not found. Please update IMAGE_PATH.")
    exit()


img = Image.open(IMAGE_PATH).convert("RGB")
x = transform(img).unsqueeze(0).to(DEVICE)


with torch.no_grad():
    outputs = model(x)
    probs = F.softmax(outputs, dim=1)


leaf_index = None
not_leaf_index = None

for i, name in enumerate(class_names):
    if name.lower() == "leaf":
        leaf_index = i
    if name.lower() in ["not_leaf", "non_leaf", "notleaf"]:
        not_leaf_index = i


if leaf_index is None or not_leaf_index is None:
    raise ValueError("Could not find Leaf / Not_Leaf labels in leaf_labels.csv")


leaf_prob = probs[0][leaf_index].item()
not_leaf_prob = probs[0][not_leaf_index].item()


predicted_idx = torch.argmax(probs, dim=1).item()
predicted_class = class_names[predicted_idx]
confidence = probs[0][predicted_idx].item()


print("\n========== LEAF DETECTOR RESULT ==========")
print(f"Image Path       : {IMAGE_PATH}")
print(f"Predicted Class  : {predicted_class}")
print(f"Confidence       : {confidence:.4f}")
print(f"Leaf Probability : {leaf_prob:.4f}")
print(f"NotLeaf Prob     : {not_leaf_prob:.4f}")


if leaf_prob >= THRESHOLD:
    print("\n✅ PASSED GATE → Send to disease model")
else:
    print("\n❌ REJECTED → Not a valid leaf image")