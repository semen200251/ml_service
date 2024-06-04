from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
from PIL import Image
from io import BytesIO
import torch
import base64
import json
import torchvision.transforms as transforms

app = FastAPI()

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class ImageRequest(BaseModel):
    taskId: int
    imageUrl: str

@app.post("/preprocess/")
async def preprocess_image(request: ImageRequest):
    try:
        response = requests.get(request.imageUrl)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")

        input_tensor = preprocess(image)
        input_tensor = input_tensor.unsqueeze(0)

        tensor_bytes = input_tensor.numpy().tobytes()
        tensor_base64 = base64.b64encode(tensor_bytes).decode('utf-8')
        
        payload = {
            "taskId": request.taskId,
            "image_tensor": tensor_base64
        }

        model_response = requests.post("http://localhost:8001/predict/", json=payload)
        model_response.raise_for_status()
        result = model_response.json()

        return result

    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Error fetching image: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
