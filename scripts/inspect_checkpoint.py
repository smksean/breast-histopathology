import sys
from app.inference import BreastHistoModel

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/inspect_checkpoint.py <model_path>")
        sys.exit(1)

    model_path = sys.argv[1]

    print(f"[INFO] Inspecting model at: {model_path}")
    model = BreastHistoModel(model_path)
    print(model.model)  # Print architecture

