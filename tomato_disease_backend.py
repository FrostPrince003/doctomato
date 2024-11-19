from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import uvicorn

# Initialize FastAPI app
app = FastAPI()

# Define device (CUDA if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the model architecture (should match the one used during training)
num_classes = 10  # Adjust this to match your dataset's number of classes
model = models.resnet18(pretrained=False)  # Use the same model architecture as during training
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Load the trained model weights (from a .pth file)
model.load_state_dict(torch.load("tomato_disease_detection.pth", map_location=device))
model = model.to(device)
model.eval()  # Set the model to evaluation mode

# Define preprocessing pipeline
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Define class names (replace with your actual class names)
class_names = [
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___healthy", "Tomato___Late_blight", "Tomato___Leaf_Mold", 
    "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot", "Tomato___Tomato_mosaic_virus", 
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus"
]

# Prediction function
def predict(image: Image.Image):
    # Preprocess the image
    image = preprocess(image).unsqueeze(0).to(device)  # Add batch dimension and move to device
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)  # Get the index of the highest probability
    return class_names[predicted.item()]

# FastAPI endpoint for prediction
@app.post("/predict/")
async def predict_endpoint(file: UploadFile = File(...)):
    try:
        # Read the uploaded image file
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")  # Convert to RGB
        
        # Make a prediction
        predicted_label = predict(image)
        
        # Return the predicted label in JSON response
        return JSONResponse(content={"predicted_label": predicted_label})
    
    except Exception as e:
        # Handle errors (e.g., invalid file format)
        return JSONResponse(content={"error": str(e)}, status_code=400)

if __name__=="__main__":
    uvicorn.run(app,host='192.168.1.105',port=8000,reload=True)