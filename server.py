import io
import numpy as np
import onnxruntime as ort
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from pathlib import Path

# Initialize app.
app = FastAPI(title="MNIST CNN Classifier")

# Load CNN at module level. Instance is shared between requests.
session = ort.InferenceSession("mnist_cnn.onnx")

# Response schema
class Prediction(BaseModel):
    digit: int
    confidence: float

# Preprocessing 
MNIST_MEAN = 0.1307
MNIST_STD  = 0.3081

def preprocess(image_bytes: bytes) -> np.ndarray:
    # Read, Resize, Normalize, Reshape
    img = Image.open(io.BytesIO(image_bytes)).convert("L")
    img = img.resize((28, 28))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - MNIST_MEAN) / MNIST_STD
    return arr[np.newaxis, np.newaxis, :, :]    # → (1, 1, 28, 28)

# Routes 
@app.get("/", response_class=HTMLResponse)
def index():
    return Path("ui.html").read_text(encoding="utf-8")

@app.post("/cnn", response_model=Prediction)
async def classify(file: UploadFile = File(...)):
    try:
        arr = preprocess(await file.read())
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    # CNN Inference
    input_name = session.get_inputs()[0].name
    logits     = session.run(None, {input_name: arr})[0]

    # CNN returns raw logits for classes 0 to 9.
    # We apply softmax here.
    exp    = np.exp(logits - logits.max())
    probs  = exp / exp.sum()
    digit  = int(probs.argmax())

    # Return respose according to schema above.
    return Prediction(digit=digit, confidence=round(float(probs[0, digit]), 4))
