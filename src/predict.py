import base64
import torch
import numpy as np
from torchvision import models
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from io import BytesIO

app = FastAPI()

model = models.resnet101()
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 27)
model.load_state_dict(torch.load('resnet101.pth', map_location=torch.device('cpu')))
model.eval()

class ImageRequest(BaseModel):
    task_id: int
    image_tensor: str

@app.post("/predict/")
async def predict(request: ImageRequest):
    try:
        tensor_bytes = base64.b64decode(request.image_tensor)
        np_array = np.frombuffer(tensor_bytes, dtype=np.float32).copy()
        input_tensor = torch.from_numpy(np_array).reshape((1, 3, 224, 224))
        with torch.no_grad():
            output = model(input_tensor)
        
        _, predicted_idx = torch.max(output, 1)
        predicted_class_id = predicted_idx.item()

        return {"task_id": request.task_id, "class_id": predicted_class_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")
    
if __name__ == "main":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
