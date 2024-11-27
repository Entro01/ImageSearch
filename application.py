import io
from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile
import aiohttp
from PIL import Image
import imagehash
from io import BytesIO
import requests
from requests.auth import HTTPBasicAuth
import json
import base64
import boto3
from typing import Optional
from pydantic import BaseModel, HttpUrl, ValidationInfo, field_validator

# Configuration Constants
OPENSEARCH_URL = "https://search-reverseimagesearch-j3nx2t2f42fy7wfayhbh3zyenq.aos.us-east-1.on.aws"
BEDROCK_MODEL_ID = "amazon.titan-embed-image-v1"
REGION = "us-east-1"
BASE_CDN_URL = "https://d1it09c4puycyh.cloudfront.net"
DEFAULT_IMAGE_DIMENSIONS = "355x503"
TITAN_IMAGE_DIMENSIONS = "448x448"

# Authentication and Clients
auth = HTTPBasicAuth('admin', '1337@Open')
bedrock_client = boto3.client(
    "bedrock-runtime", 
    REGION, 
    endpoint_url=f"https://bedrock-runtime.{REGION}.amazonaws.com"
)

app = FastAPI(
    title="Reverse Image Search API",
    description="API for finding similar images using perceptual hashing and vector embeddings"
)

async def fetch_image(url: str) -> bytes:
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status == 200:
                return await response.read()
            else:
                raise HTTPException(status_code=404, detail="Image not found")

def calculate_phash(img):
    long_side = max(img.size)
    ratio = 512 / long_side
    new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
    img = img.resize(new_size, Image.Resampling.LANCZOS)
    new_img = Image.new('RGB', (512, 512), (255, 255, 255))
    paste_pos = ((512 - new_size[0]) // 2, (512 - new_size[1]) // 2)
    new_img.paste(img, paste_pos)
    
    # converting phash to binary
    phash = imagehash.phash(new_img)
    
    # imageHash to integer and then to binary string
    phash_int = int(str(phash), 16)  # converting pHash to a hexadecimal integer
    binary_string = f"{phash_int:0>64b}"  # converting to a 64-bit binary string
    
    # converting the binary string to a list of binary integers
    return [int(bit) for bit in binary_string]

@app.get("/find_same/")
async def find_similar(
    image_url: str = Query(..., description="URL of the image to find similar items for"),
    top: int = Query(1, description="Number of similar URLs to return")
):
    try:
        img = await fetch_image(image_url)
        input_phash = calculate_phash(img)
        
        search_results = query_opensearch(input_phash, top_n=top, index_type='phash')
        
        results = []
        for result in search_results:
            image_url = f"{BASE_CDN_URL}/{DEFAULT_IMAGE_DIMENSIONS}/catalog/product{result['_source']['small_image'].strip()}"
            results.append({
                "entity_id": result["_source"]["entity_id"],
                "sku": result["_source"]["sku"],
                "image_url": image_url,
                "score": result["_score"]  # KNN similarity score
            })
        
        return {"matches": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


### Reverse Image Search

def create_image_embedding(image_base64: str) -> list:
    """
    Generate image embedding using Amazon Titan model.
    
    Args:
        image_base64 (str): Base64 encoded image
    
    Returns:
        list: Image embedding vector
    
    Raises:
        HTTPException: If embedding generation fails
    """
    if not image_base64:
        raise HTTPException(status_code=400, detail="Invalid image")
    
    image_input = {"inputImage": image_base64}
    
    try:
        bedrock_response = bedrock_client.invoke_model(
            body=json.dumps(image_input),
            modelId=BEDROCK_MODEL_ID,
            accept="application/json",
            contentType="application/json"
        )
        
        final_response = json.loads(bedrock_response.get("body").read())
        
        # Check for any errors in the response
        if "message" in final_response:
            raise HTTPException(status_code=400, detail=f"Embedding error: {final_response['message']}")
        
        return final_response.get("embedding")
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Embedding generation failed: {str(e)}")

def query_opensearch(embedding: list, top_n: int = 1, index_type: str = 'vector') -> list:
    """
    Query OpenSearch with vector or perceptual hash embedding.
    
    Args:
        embedding (list): Image embedding or perceptual hash
        top_n (int, optional): Number of top results to return. Defaults to 1.
        index_type (str, optional): Type of index to query. Defaults to 'vector'.
    
    Returns:
        list: Search results from OpenSearch
    
    Raises:
        HTTPException: If OpenSearch query fails
    """
    # Construct query based on index type
    query_map = {
        'phash': {
            "size": top_n,
            "query": {
                "knn": {
                    "phash": {
                        "vector": embedding,
                        "k": top_n
                    }
                }
            }
        },
        'vector': {
            "size": top_n,
            "query": {
                "knn": {
                    "vector": {
                        "vector": embedding,
                        "k": top_n
                    }
                }
            },
            "_source": ["product_id"]
        }
    }
    
    if index_type not in query_map:
        raise ValueError(f"Invalid index type: {index_type}")
    
    # Perform the search
    try:
        response = requests.get(
            f"{OPENSEARCH_URL}/_search", 
            json=query_map[index_type], 
            auth=auth
        )
        
        response.raise_for_status()
        return response.json()['hits']['hits']
    
    except requests.RequestException as e:
        raise HTTPException(
            status_code=500, 
            detail=f"OpenSearch query failed: {str(e)}"
        )
    
@app.post("/find_similar/")
async def find_similar_images(
    image_url: Optional[str] = Query(None, description="URL of the image to find similar items for"),
    image: Optional[UploadFile] = File(None, description="Image file to find similar items for"),
    top: int = Query(1, description="Number of similar URLs to return")
):
    """
    Unified endpoint for finding similar images via URL or file upload.
   
    Prioritizes image URL over file upload if both are provided.
   
    Args:
        image_url (Optional[str]): URL of the image to search
        image (Optional[UploadFile]): File upload of the image to search
        top (int): Number of top similar results to return
   
    Returns:
        dict: Matching product information with similarity scores
    """
    # Prioritize URL if both URL and image are provided
    if image_url:
        try:
            # Fetch image from URL
            contents = await fetch_image(image_url)
            print(contents)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to fetch image from URL: {str(e)}"
            )
    elif image:
        # Process uploaded file
        try:
            contents = await image.read()
            print(contents)
            img = Image.open(io.BytesIO(contents))
            img.verify()
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to process uploaded image: {str(e)}"
            )
    else:
        # No image source provided
        raise HTTPException(
            status_code=400,
            detail="Either image URL or image file must be provided"
        )
    
    try:
        # Optional: Validate image (convert to PIL Image to check)
        Image.open(BytesIO(contents)).convert('RGB')
        
        # Preprocess image
        base64_image = base64.b64encode(contents).decode('utf-8')
       
        # Generate embedding
        embedding = create_image_embedding(base64_image)
       
        # Search similar images
        search_results = query_opensearch(
            embedding,
            top_n=top,
            index_type='vector'
        )
       
        # Format results
        results = [
            {
                "product_id": result["_source"]["product_id"],
                "score": result["_score"]
            } for result in search_results
        ]
       
        return {"matches": results}
   
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image search failed: {str(e)}")

@app.get("/")
async def root():
    """
    Health check endpoint.
    """
    return {"message": "Reverse Image Search API is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)