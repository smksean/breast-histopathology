from PIL import Image
from torchvision import transforms
import torch

# --------------------
# SETTINGS
# --------------------
# Class names for predictions (must match training order)
CLASS_NAMES = ["benign", "malignant"]

# --------------------
# IMAGE PREPROCESSING
# --------------------
def get_transform(input_size: int = 224):
    """
    Returns a torchvision transform that:
    - Resizes to a fixed square size (input_size x input_size)
    - Converts to tensor
    - Normalizes with ImageNet stats (must match training)
    """
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet mean
            std=[0.229, 0.224, 0.225]    # ImageNet std
        )
    ])

def load_image_as_tensor(path: str, input_size: int = 224) -> torch.Tensor:
    """
    Loads an image from path, applies preprocessing, 
    and adds a batch dimension [1, C, H, W].
    """
    img = Image.open(path).convert("RGB")
    transform = get_transform(input_size)
    return transform(img).unsqueeze(0)  # batch of size 1

