import torch
import torch.nn as nn
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import models, transforms
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "../best_leaf_detector_model.pth"


def load_trained_model():

    print("Loading model...")

    model = models.efficientnet_b0(weights=None)

    in_features = model.classifier[1].in_features

    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.4),
        nn.Linear(256, 2)
    )

    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)

    model.eval()
    model.to(DEVICE)

    print("Model loaded successfully.")
    return model


class GradCAM:

    def __init__(self, model, target_layer):

        self.model = model
        self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        self.forward_handle = target_layer.register_forward_hook(self.save_activations)
        self.backward_handle = target_layer.register_full_backward_hook(self.save_gradients)

    def save_activations(self, module, input, output):
        self.activations = output.detach()

    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def remove_hooks(self):
        self.forward_handle.remove()
        self.backward_handle.remove()

    def generate_heatmap(self, input_tensor, class_idx):

        self.model.zero_grad()

        output = self.model(input_tensor)
        loss = output[0, class_idx]

        print("Running backward pass...")
        loss.backward()
        print("Backward pass complete.")

        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)

        heatmap = torch.sum(weights * self.activations, dim=1).squeeze()

        heatmap = heatmap.cpu().numpy()
        heatmap = np.maximum(heatmap, 0)

        heatmap = cv2.resize(heatmap, (224, 224))

        denom = heatmap.max() - heatmap.min()
        if denom != 0:
            heatmap = (heatmap - heatmap.min()) / denom

        return heatmap


def run_test(image_path):

    model = load_trained_model()

    # EfficientNet last conv layer
    target_layer = model.features[-1]

    cam = GradCAM(model, target_layer)

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    print("Loading image...")
    img_pil = Image.open(image_path).convert("RGB")

    input_tensor = preprocess(img_pil).unsqueeze(0).to(DEVICE)

    print("Running inference...")
    with torch.no_grad():
        output = model(input_tensor)
        class_idx = torch.argmax(output).item()

    print("Predicted class:", class_idx)

    print("Generating Grad-CAM heatmap...")
    heatmap = cam.generate_heatmap(input_tensor, class_idx)

    cam.remove_hooks()

    print("Loading image with OpenCV...")
    img_cv = cv2.imread(image_path)

    if img_cv is None:
        raise ValueError("OpenCV failed to load the image.")

    img_cv = cv2.resize(img_cv, (224, 224))

    heatmap_colored = cv2.applyColorMap(
        np.uint8(255 * heatmap),
        cv2.COLORMAP_JET
    )

    superimposed_img = heatmap_colored * 0.4 + img_cv * 0.6
    superimposed_img = np.uint8(superimposed_img)

    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))

    plt.title(
        f"Prediction: {'LEAF' if class_idx == 0 else 'NON-LEAF'}"
    )

    plt.axis("off")

    plt.show(block=False)
    plt.pause(10)
    plt.close()
    print("Grad-CAM visualization complete.")


if __name__ == "__main__":

    run_test(
        r"C:\Users\Accer\Downloads\world-of-warcraft-dragon-fire-face-wallpaper-preview.jpg"
    )