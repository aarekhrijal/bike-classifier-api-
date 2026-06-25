from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import torch
import torchvision.transforms as transforms
from fastai.vision.all import load_learner, PILImage
from PIL import Image
import io
import pathlib

app = FastAPI()

# Fix Windows path issue
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
learn = load_learner("bike_classifier.pkl")
pathlib.PosixPath = temp

# Get class names from the model
class_names = learn.dls.vocab

# Standard imagenet transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.get("/")
def home():
    return {"message": "Bike Brand Classifier API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        output = learn.model(tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        confidence, predicted = torch.max(probabilities, 0)
    
    return JSONResponse({
        "predicted_class": str(class_names[predicted.item()]),
        "confidence": round(confidence.item(), 4)
    })