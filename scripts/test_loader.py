import argparse
import os
from glob import glob
from app.inference import BreastHistoModel

def collect_images(images_dir: str | None, images: list[str] | None):
    paths = []
    if images_dir:
        for ext in ("*.png","*.jpg","*.jpeg","*.tif","*.tiff","*.bmp","*.webp"):
            paths.extend(glob(os.path.join(images_dir, ext)))
    if images:
        paths.extend(images)
    # unique, keep order
    seen = set()
    unique = []
    for p in paths:
        if p not in seen:
            unique.append(p)
            seen.add(p)
    return unique

def main():
    ap = argparse.ArgumentParser(description="Test model loader and aggregate predictions.")
    ap.add_argument("--model", required=True, help="Path to .pt file")
    ap.add_argument("--images_dir", help="Folder containing patches (same slide)")
    ap.add_argument("--image", action="append", help="Add one image path (repeatable)")
    ap.add_argument("--aggregate", default="mean", choices=["mean","median","vote"], help="Aggregation method")
    ap.add_argument("--input_size", type=int, default=224)
    args = ap.parse_args()

    model = BreastHistoModel(model_path=args.model, input_size=args.input_size, device="cpu")

    # SINGLE IMAGE test (if only one image provided)
    if args.image and not args.images_dir and len(args.image) == 1:
        res = model.predict_single(args.image[0])
        print("\n=== single image prediction ===")
        print(res)
        return

    # COLLECT multiple patches
    image_paths = collect_images(args.images_dir, args.image)

    if not image_paths:
        print("No images found. Provide --images_dir or --image ...")
        return

    print(f"Found {len(image_paths)} images. Running aggregate='{args.aggregate}'...")
    res = model.predict_aggregate(image_paths, aggregate=args.aggregate)
    print("\n=== aggregate prediction ===")
    print(f"aggregate method : {res['aggregate']}")
    print(f"num patches      : {res['num_patches']}")
    print(f"probabilities    : {res['probabilities']}")
    print(f"predicted label  : {res['predicted_label']} (index {res['predicted_index']})")

if __name__ == "__main__":
    main()
