import os
import json
import gdown

import gradio as gr
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms
import matplotlib.pyplot as plt


# PATHS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "best_model_stage2.pth")
CLASS_MAPPING_PATH = os.path.join(BASE_DIR, "class_mapping.json")

MODEL_DRIVE_FILE_ID = "15b4vCoXIQ87AX-bpM5Ct7MKarZM1LiV6"
MODEL_DRIVE_URL = f"https://drive.google.com/file/d/15b4vCoXIQ87AX-bpM5Ct7MKarZM1LiV6/view?usp=sharing"

if not os.path.exists(MODEL_PATH):
    print("Model checkpoint not found. Downloading from Google Drive...")
    gdown.download(MODEL_DRIVE_URL, MODEL_PATH, quiet=False)
    print("Model checkpoint downloaded successfully.")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("best_model_stage2.pth was not found.")

model_size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
print(f"Using model file: {os.path.abspath(MODEL_PATH)}")
print(f"Model file size: {model_size_mb:.2f} MB")

if model_size_mb < 10:
    raise ValueError(
        "The model file seems too small. It may not be the real .pth checkpoint. "
        "Please re-download best_model_stage2.pth from Google Drive."
    )


# DEVICE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# LOAD CLASS MAPPING
# No hardcoded prediction order.
# Order is taken from class_mapping.json.

if not os.path.exists(CLASS_MAPPING_PATH):
    raise FileNotFoundError(
        "class_mapping.json was not found. "
        "Please place it in the same folder as app.py."
    )

with open(CLASS_MAPPING_PATH, "r") as f:
    class_mapping = json.load(f)


def get_class_names_from_mapping(mapping):
    """
    Reads class names in the exact index order used during training.
    Priority:
    1. class_to_idx
    2. idx_to_class
    3. classes
    """

    if "class_to_idx" in mapping:
        class_to_idx = mapping["class_to_idx"]
        ordered_pairs = sorted(
            class_to_idx.items(),
            key=lambda item: int(item[1])
        )
        return [class_name for class_name, idx in ordered_pairs]

    if "idx_to_class" in mapping:
        idx_to_class = mapping["idx_to_class"]
        ordered_indices = sorted(
            idx_to_class.keys(),
            key=lambda index: int(index)
        )
        return [idx_to_class[index] for index in ordered_indices]

    if "classes" in mapping:
        return mapping["classes"]

    raise ValueError(
        "class_mapping.json does not contain class_to_idx, idx_to_class, or classes."
    )


class_names = get_class_names_from_mapping(class_mapping)
NUM_CLASSES = len(class_names)
print("Class mapping loaded from class_mapping.json:")
for i, name in enumerate(class_names):
    print(f"{i}: {name}")

print("Loaded class names from class_mapping.json:")
for idx, name in enumerate(class_names):
    print(f"{idx}: {name}")


def clean_display_name(class_key):
    return class_key.replace("_", " ").title()


alphabetical_display_order = sorted(class_names, key=clean_display_name)


# IMAGE TRANSFORM
# Same preprocessing used during evaluation

image_transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# MODEL CREATION

def create_model():
    model = models.efficientnet_b3(weights=None)

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(1536, 512),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(512, NUM_CLASSES)
    )

    return model


model = create_model()

checkpoint = torch.load(MODEL_PATH, map_location=device)

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    state_dict = checkpoint["model_state_dict"]
else:
    state_dict = checkpoint

model.load_state_dict(state_dict, strict=True)

model = model.to(device)
model.eval()

print("Model loaded successfully.")


# GRAD-CAM FUNCTION

def generate_gradcam(original_img, predicted_idx):
    model.eval()

    display_size = (340, 340)

    original_rgb = original_img.convert("RGB")
    input_tensor = image_transform(original_rgb).unsqueeze(0).to(device)

    target_layer = model.features[-1]

    activations = {}
    gradients = {}

    def forward_hook(module, input, output):
        activations["value"] = output

    def backward_hook(module, grad_input, grad_output):
        gradients["value"] = grad_output[0]

    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)

    model.zero_grad()

    outputs = model(input_tensor)
    class_score = outputs[0, predicted_idx]
    class_score.backward()

    forward_handle.remove()
    backward_handle.remove()

    activation_maps = activations["value"]
    gradient_maps = gradients["value"]

    weights = torch.mean(gradient_maps, dim=(2, 3), keepdim=True)

    cam = torch.sum(weights * activation_maps, dim=1)
    cam = F.relu(cam)

    cam = F.interpolate(
        cam.unsqueeze(1),
        size=(300, 300),
        mode="bilinear",
        align_corners=False
    )

    cam = cam.squeeze().detach().cpu().numpy()

    cam = cam - cam.min()
    if cam.max() != 0:
        cam = cam / cam.max()

    heatmap = plt.get_cmap("jet")(cam)[:, :, :3]

    original_display = original_rgb.resize(display_size)
    original_array = np.array(original_display).astype(np.float32) / 255.0

    heatmap_img = Image.fromarray((heatmap * 255).astype(np.uint8)).resize(display_size)
    heatmap_array = np.array(heatmap_img).astype(np.float32) / 255.0

    overlay = 0.45 * heatmap_array + 0.55 * original_array
    overlay = np.clip(overlay, 0, 1)

    overlay_img = Image.fromarray((overlay * 255).astype(np.uint8))

    return overlay_img


# FILE UPLOAD STATE FUNCTION

def store_uploaded_file(file):
    if file is None:
        return None, "<div class='upload-status'>No image selected</div>"

    if isinstance(file, str):
        file_path = file
    elif hasattr(file, "path"):
        file_path = file.path
    elif hasattr(file, "name"):
        file_path = file.name
    else:
        file_path = str(file)

    file_name = os.path.basename(file_path)

    return file_path, f"<div class='upload-status selected'>Selected: {file_name}</div>"


# PREDICTION FUNCTION

def predict_image(image_path):
    if image_path is None:
        empty_html = """
        <div class="result-text">
            <div class="section-label">Predicted Class</div>
            <div class="empty-message">Upload an image first.</div>
        </div>
        """
        return empty_html, None

    try:
        original_img = Image.open(image_path).convert("RGB")
    except Exception:
        error_html = """
        <div class="result-text">
            <div class="section-label">Predicted Class</div>
            <div class="empty-message">Could not read the uploaded file.</div>
        </div>
        """
        return error_html, None

    input_tensor = image_transform(original_img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
    
    print("\nPrediction debug:")
    print("Uploaded image path:", image_path)

    print("Raw logits:")
    for i, value in enumerate(outputs[0].detach().cpu().numpy()):
        print(f"{i} | {class_names[i]} | logit: {value:.4f}")

    print("Probabilities:")
    for i, prob in enumerate(probabilities.detach().cpu().numpy()):
        print(f"{i} | {class_names[i]} | probability: {prob * 100:.2f}%")
    

    predicted_idx = torch.argmax(probabilities).item()
    predicted_class_key = class_names[predicted_idx]
    predicted_display_name = clean_display_name(predicted_class_key)
    confidence = probabilities[predicted_idx].item()

    # Terminal debug print
    print("\nPrediction debug")
    print(f"Image path: {image_path}")
    print(f"Predicted index: {predicted_idx}")
    print(f"Predicted key: {predicted_class_key}")
    print("All probabilities:")

    ranked_indices = torch.argsort(probabilities, descending=True).cpu().tolist()

    for idx in ranked_indices:
        print(f"{idx} | {clean_display_name(class_names[idx])}: {probabilities[idx].item() * 100:.2f}%")

    gradcam_img = generate_gradcam(original_img, predicted_idx)

    probability_lines = ""

    for class_key in alphabetical_display_order:
        idx = class_names.index(class_key)
        class_display = clean_display_name(class_key)
        prob = probabilities[idx].item() * 100

        probability_lines += f"""
        <div class="prob-row">
            <span>{class_display}</span>
            <span>{prob:.2f}%</span>
        </div>
        """

    result_html = f"""
    <div class="result-text">
        <div class="section-label">Predicted Class</div>
        <div class="prediction-main">{predicted_display_name}</div>
        <div class="confidence-main">Confidence: {confidence * 100:.2f}%</div>

        <div class="section-label score-heading">Confidence Scores</div>
        <div class="prob-box">
            {probability_lines}
        </div>
    </div>
    """

    return result_html, gradcam_img


# CUSTOM CSS

custom_css = """
body {
    background: #061522 !important;
    color: #F5FAFF !important;
}

.gradio-container {
    max-width: 980px !important;
    margin: auto !important;
    background: #061522 !important;
    color: #F5FAFF !important;
    font-family: Inter, Arial, sans-serif !important;
}

#title-block {
    text-align: center;
    margin-top: 28px;
    margin-bottom: 22px;
}

#title-block h1 {
    font-size: 38px;
    font-weight: 850;
    letter-spacing: -1px;
    color: #FFFFFF;
    margin: 0;
}

#main-panel {
    max-width: 860px;
    margin: 0 auto 28px auto;
    background: #0B1E31;
    border: 1px solid #1F4A68;
    border-radius: 18px;
    padding: 26px 28px;
    box-shadow: 0 12px 32px rgba(0, 0, 0, 0.30);
}

#upload-strip {
    max-width: 540px;
    margin: 0 auto;
}

#upload-button {
    width: 100% !important;
}

#upload-button button {
    width: 100% !important;
    background: #C8D3DD !important;
    color: #061522 !important;
    border: none !important;
    border-radius: 8px !important;
    min-height: 34px !important;
    padding: 7px 10px !important;
    font-size: 13px !important;
    font-weight: 800 !important;
}

#upload-button button:hover {
    background: #E0E7EC !important;
}

.upload-status {
    text-align: center;
    color: #8DB7D0;
    font-size: 11px;
    margin-top: 8px;
    margin-bottom: 0;
}

.upload-status.selected {
    color: #B7D5E8;
}

#predict-button {
    max-width: 540px;
    margin: 14px auto 24px auto;
}

#predict-button button {
    width: 100%;
    background: #C8D3DD !important;
    color: #061522 !important;
    border: none !important;
    border-radius: 8px !important;
    font-size: 12px !important;
    letter-spacing: 1.8px !important;
    font-weight: 850 !important;
    padding: 7px 10px !important;
    min-height: 34px !important;
    text-transform: uppercase;
}

#predict-button button:hover {
    background: #E0E7EC !important;
}

#result-box {
    max-width: 780px;
    margin: 0 auto;
    background: #081A2B;
    border: 1px solid #2B5B7A;
    border-radius: 15px;
    padding: 18px;
}

.result-text {
    color: #F5FAFF;
    padding: 2px 8px;
}

.section-label {
    font-size: 10.5px;
    color: #8EC5E8;
    font-weight: 850;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    margin-bottom: 6px;
}

.prediction-main {
    font-size: 22px;
    font-weight: 850;
    color: #FFFFFF;
    margin-bottom: 5px;
    line-height: 1.12;
}

.confidence-main {
    font-size: 12px;
    color: #B7D5E8;
    margin-bottom: 12px;
}

.score-heading {
    margin-top: 9px !important;
    margin-bottom: 5px !important;
}

.prob-box {
    width: 100%;
}

.prob-row {
    display: flex;
    justify-content: space-between;
    border-bottom: 1px solid #1B405A;
    padding: 5px 0;
    font-size: 11.5px;
    color: #EAF6FF;
}

.empty-message {
    color: #9EBBD0;
    font-size: 12px;
}

#gradcam-image {
    display: flex;
    align-items: center;
    justify-content: center;
}

#gradcam-image label {
    display: none !important;
}

#gradcam-image img {
    width: 330px !important;
    height: 330px !important;
    object-fit: contain !important;
    border-radius: 10px !important;
    border: 1px solid #2B5B7A !important;
    background: #061522 !important;
}

.block {
    background: transparent !important;
    border: none !important;
}

footer {
    display: none !important;
}
"""


# GRADIO INTERFACE

with gr.Blocks(css=custom_css, theme=gr.themes.Base()) as demo:

    gr.HTML(
        """
        <div id="title-block">
            <h1>CosmicVision</h1>
        </div>
        """
    )

    uploaded_image_state = gr.State(value=None)

    with gr.Column(elem_id="main-panel"):

        with gr.Column(elem_id="upload-strip"):
            upload_button = gr.UploadButton(
                "Upload Image",
                file_types=["image"],
                file_count="single",
                elem_id="upload-button"
            )

            upload_status = gr.HTML(
                """
                <div class="upload-status">No image selected</div>
                """
            )

        upload_button.upload(
            fn=store_uploaded_file,
            inputs=upload_button,
            outputs=[uploaded_image_state, upload_status]
        )

        with gr.Column(elem_id="predict-button"):
            predict_button = gr.Button("Predict")

        with gr.Row(elem_id="result-box"):
            with gr.Column(scale=44):
                result_output = gr.HTML(
                    """
                    <div class="result-text">
                        <div class="section-label">Predicted Class</div>
                        <div class="empty-message">Upload an image and click Predict.</div>
                    </div>
                    """
                )

            with gr.Column(scale=56):
                gradcam_output = gr.Image(
                    label=None,
                    show_label=False,
                    type="pil",
                    elem_id="gradcam-image"
                )

        predict_button.click(
            fn=predict_image,
            inputs=uploaded_image_state,
            outputs=[result_output, gradcam_output]
        )


if __name__ == "__main__":
    demo.launch()