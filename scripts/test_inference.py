import torch
import os
from PIL import Image
from torchvision import transforms

# --------------------
# SETTINGS
# --------------------
MODEL_PATH = "models/resnet50_bh_e1_ts_probs.pt"  # already softmaxed
CLASS_NAMES = ["benign", "malignant"]
IMAGE_PATH = "images/malignant"  # folder or single image

# --------------------
# LOAD MODEL
# --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.jit.load(MODEL_PATH, map_location=device)
model.eval()

# --------------------
# TRANSFORMS — must match training exactly
# --------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # same as in training loader
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# --------------------
# PREDICT ONE IMAGE
# --------------------
def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = model(img_tensor)[0]  # shape: [num_classes]
    malignant_prob = probs[1].item()
    pred_bin = int(malignant_prob >= 0.5)  # match training cutoff
    pred_label = CLASS_NAMES[pred_bin]
    return probs.cpu().tolist(), pred_bin, pred_label

# --------------------
# FOLDER MODE — replicate val loop logic
# --------------------
def predict_folder(folder_path):
    all_preds = []
    all_probs = []
    for fname in os.listdir(folder_path):
        if fname.lower().endswith((".png", ".jpg", ".jpeg")):
            fpath = os.path.join(folder_path, fname)
            probs, pred_bin, _ = predict_image(fpath)
            all_preds.append(pred_bin)
            all_probs.append(probs)
    if not all_preds:
        raise ValueError("No valid images found.")
    # Majority vote like val loop, not avg prob
    final_pred_bin = int(sum(all_preds) >= len(all_preds) / 2)
    final_label = CLASS_NAMES[final_pred_bin]
    avg_probs = torch.tensor(all_probs).mean(dim=0).tolist()
    return avg_probs, final_pred_bin, final_label

# --------------------
# MAIN
# --------------------
if os.path.isdir(IMAGE_PATH):
    print(f"[INFO] Folder detected: {IMAGE_PATH}")
    avg_probs, pred_bin, pred_label = predict_folder(IMAGE_PATH)
    print(f"[FINAL AGGREGATE] avg probs: {avg_probs}, predicted class: {pred_label} (bin: {pred_bin})")
else:
    probs, pred_bin, pred_label = predict_image(IMAGE_PATH)
    print(f"probs: {probs}, predicted class: {pred_label} (bin: {pred_bin})")



