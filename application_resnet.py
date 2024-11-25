from fastapi import FastAPI, HTTPException, Query
import torch
from torchvision import models, transforms
from torchvision.transforms.functional import resize, center_crop
from PIL import Image
import aiohttp
import numpy as np
from io import BytesIO
import boto3
import json
from sklearn.neighbors import NearestNeighbors

app = FastAPI()

dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
table = dynamodb.Table('LuluFeatureStore')

model = models.resnet50(pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1])
model.eval()

preprocess = transforms.Compose([
    transforms.Lambda(lambda img: resize(img, size=256)),  # resizing to 256 while maintaining aspect ratio
    transforms.Lambda(lambda img: center_crop(img, 224)),  # center cropping to 224x224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

async def fetch_image(url: str):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status == 200:
                img_data = await response.read()
                return Image.open(BytesIO(img_data)).convert('RGB')
            else:
                raise HTTPException(status_code=404, detail="Image not found")

async def extract_features(image_url: str):
    img = await fetch_image(image_url)
    img_tensor = preprocess(img).unsqueeze(0)
    with torch.no_grad():
        features = model(img_tensor).flatten().numpy()
    return features

def find_nearest_neighbors(feature_vector, all_features, top_n=3):
    nn_model = NearestNeighbors(n_neighbors=top_n, metric='euclidean')
    nn_model.fit(all_features)
    distances, indices = nn_model.kneighbors([feature_vector])
    return indices[0], distances[0]

def calculate_match_percentage(distance, max_distance):
    return max(0, 100 * (1 - distance / max_distance))

def load_all_features():
    response = table.scan()
    items = response['Items']
    features = [json.loads(item['features']) for item in items]
    return items, np.array(features)

@app.get("/find_similar/")
async def find_similar(image_url: str = Query(..., description="URL of the image to find similar items for")):
    try:
        input_features = await extract_features(image_url)
        all_items, all_features = load_all_features()
        indices, distances = find_nearest_neighbors(input_features, all_features)
        max_distance = np.linalg.norm(np.ones(2048))  # max distance between two 2048-dimension vectors

        base_url = "https://d1it09c4puycyh.cloudfront.net"
        dimensions = "355x503"

        results = []
        for idx, i in enumerate(indices):
            percentage_match = calculate_match_percentage(distances[idx], max_distance)
            image_url = f"{base_url}/{dimensions}/catalog/product{all_items[i]['small_image'].strip()}"
            results.append({
                "entity_id": all_items[i]["entity_id"],
                "sku": all_items[i]["sku"],
                "image_url": image_url,
                "match_percentage": percentage_match
            })

        return {"matches": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
