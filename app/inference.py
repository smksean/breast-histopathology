import torch
import torchvision.transforms as transforms
from PIL import Image
import os

class BreastHistoModel:
    def __init__(self, model_path, class_names=None):
        # Default to benign/malignant if no custom names provided
        self.class_names = class_names or ["benign", "malignant"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model
        self.model = torch.load(model_path, map_location=self.device)
        self.model.eval()

        # Preprocessing (must match training)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet mean
                std=[0.229, 0.224, 0.225]    # ImageNet std
            )
        ])

    def predict(self, img_tensor, return_label=False):
        """Predict class index and probability for a given image tensor."""
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            pred_idx = torch.argmax(probs).item()
            pred_label = self.class_names[pred_idx]

        if return_label:
            return probs.cpu().numpy(), pred_idx, pred_label
        return probs.cpu().numpy(), pred_idx

    def predict_folder(self, folder_path, return_label=False):
        """Predict classes for all images in a folder and return aggregated result."""
        all_probs = []

        for img_name in os.listdir(folder_path):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(folder_path, img_name)
                image = Image.open(img_path).convert("RGB")
                img_tensor = self.transform(image).unsqueeze(0).to(self.device)
                probs, _ = self.predict(img_tensor, return_label=False)
                all_probs.append(torch.tensor(probs))

        if not all_probs:
            raise ValueError("No valid image files found in folder.")

        # Aggregate probabilities by mean
        avg_probs = torch.mean(torch.stack(all_probs), dim=0)
        pred_idx = torch.argmax(avg_probs).item()
        pred_label = self.class_names[pred_idx]

        if return_label:
            return avg_probs.cpu().numpy(), pred_idx, pred_label
        return avg_probs.cpu().numpy(), pred_idx




