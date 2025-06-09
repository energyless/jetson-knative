# standard libs
import os
import base64
from PIL import Image
from io import BytesIO

# request handling 
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from starlette.responses import JSONResponse


# model handling
import torch
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms


if os.environ["USE_GPU"] == "true":
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"using device {device}")


class ImagePredictionRequest(BaseModel):
    image: str
    filename: str


app = FastAPI()


weights = ResNet50_Weights.IMAGENET1K_V1
model = resnet50(weights=weights).to(device)
model.eval()


preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


with open("imagenet_classes.txt") as f:
    classes = [line.strip() for line in f]


@app.get("/health")
async def health_check():
    return JSONResponse(content={"status": "Ok"}, status_code=200)



@app.post("/infer")
async def infer(req: ImagePredictionRequest):
    try:
        img_data = base64.b64decode(req.image)
        img = Image.open(BytesIO(img_data)).convert("RGB")
        input_tensor = preprocess(img).unsqueeze(0)

        with torch.no_grad():
            outputs = model(input_tensor)
        
        probs = torch.nn.functional.softmax(outputs[0], dim=0)
        top3_prob, top3_catid = torch.topk(probs, 3)

        predictions = [
            {
                "class_id": str(top3_catid[i].item()),
                "label": classes[top3_catid[i].item()],
                "score": top3_prob[i].item()
            }
            for i in range(top3_prob.size(0))
        ]

        return JSONResponse(content={"predictions": predictions}, status_code=200)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)



def main():
    uvicorn.run(app, host="0.0.0.0", port="8080")



if __name__ == "__main__":
    main()