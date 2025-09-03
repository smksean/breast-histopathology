from fastapi import FastAPI, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from app.inference import BreastHistoModel
from PIL import Image
import io, os, torch

MODEL_PATH = os.getenv("MODEL_PATH", "models/resnet50_bh_e1_ts_probs.pt")
CLASS_NAMES = ["benign", "malignant"]

app = FastAPI(title="Breast Histo API", version="1.0")

# relax for dev; later set this to your Vercel/Netlify URL(s)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# load ONCE at startup (like Streamlit cache)
model = BreastHistoModel(model_path=MODEL_PATH, class_names=CLASS_NAMES)

@app.get("/")
def root():
    return {"name": "Breast Histo API", "classes": CLASS_NAMES, "disclaimer": "Research use only"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    raw = await file.read()
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    # use the SAME transform + device path as your working Streamlit app
    img_tensor = model.transform(img).unsqueeze(0).to(model.device)
    probs, idx, label = model.predict(img_tensor, return_label=True)
    return {
        "probabilities": probs.tolist(),  # numpy -> JSON list
        "predicted_index": int(idx),
        "predicted_label": label
    }

@app.post("/predict-batch")
async def predict_batch(files: list[UploadFile] = File(...),
                        aggregate: str = Query("mean", enum=["mean","median","vote"])):
    all_probs, votes = [], []
    for f in files:
        raw = await f.read()
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        img_tensor = model.transform(img).unsqueeze(0).to(model.device)
        probs, idx = model.predict(img_tensor, return_label=False)
        all_probs.append(torch.tensor(probs))
        votes.append(int(idx))

    if not all_probs:
        return {"detail": "No valid images."}

    stack = torch.stack(all_probs)
    if aggregate == "median":
        avg_probs = torch.median(stack, dim=0).values
        final_idx = int(torch.argmax(avg_probs).item())
    elif aggregate == "vote":
        final_idx = 1 if votes.count(1) >= len(votes)/2 else 0
        avg_probs = torch.mean(stack, dim=0)
    else:
        avg_probs = torch.mean(stack, dim=0)
        final_idx = int(torch.argmax(avg_probs).item())

    return {
        "aggregate": aggregate,
        "num_patches": len(all_probs),
        "probabilities": avg_probs.tolist(),
        "predicted_index": final_idx,
        "predicted_label": CLASS_NAMES[final_idx]
    }
