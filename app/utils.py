from PIL import Image
from torchvision import transforms
import torch

# image preprocessing – must match your training pipeline
def get_transform(input_size: int = 224):
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet
            std=[0.229, 0.224, 0.225]
        )
    ])

def load_image_as_tensor(path: str, input_size: int = 224) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    transform = get_transform(input_size)
    return transform(img).unsqueeze(0)  # shape: [1, C, H, W]
