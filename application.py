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
from tenacity import RetryError, retry, stop_after_attempt, wait_exponential

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
    
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    reraise=True
)
async def create_image_embedding_with_retry(image_base64: str) -> list:
    """
    Retry wrapper for create_image_embedding function
    """
    try:
        embedding = create_image_embedding(image_base64)
        # logger.info("Successfully created embedding")
        return embedding
    except Exception as e:
        # logger.error(f"Error creating embedding: {str(e)}")
        raise

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

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    reraise=True
)
async def query_opensearch_with_retry(embedding: list, top_n: int, index_type: str = 'vector') -> list:
    """
    Retry wrapper for query_opensearch function
    """
    try:
        results = query_opensearch(embedding, top_n=top_n, index_type=index_type)
        # logger.info("Successfully queried OpenSearch")
        return results
    except Exception as e:
        # logger.error(f"Error querying OpenSearch: {str(e)}")
        raise

async def validate_and_resize_image(image_bytes: bytes, max_pixels: int = 1024*1024) -> bytes:
    """
    Validates image size and resizes if necessary.
    
    Args:
        image_bytes: Original image bytes
        max_pixels: Maximum allowed pixels (width * height)
    
    Returns:
        bytes: Processed image bytes
    """
    img = Image.open(io.BytesIO(image_bytes))
    width, height = img.size
    total_pixels = width * height
    
    if total_pixels > max_pixels:
        # Calculate new dimensions while maintaining aspect ratio
        ratio = (max_pixels / total_pixels) ** 0.5
        new_width = int(width * ratio)
        new_height = int(height * ratio)
        
        # Resize image
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Convert back to bytes
        img_byte_arr = io.BytesIO()
        img = img.convert('RGB')  # Ensure RGB format
        img.save(img_byte_arr, format='JPEG', quality=85)
        return img_byte_arr.getvalue()
    
    return image_bytes
    
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
            print(contents[:5])
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to fetch image from URL: {str(e)}"
            )
    elif image:
        # Process uploaded file
        try:
            contents = await image.read()
            print(contents[:5])
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
        # Validate and resize image if needed
        contents = await validate_and_resize_image(contents)
        
        # Optional: Validate image (convert to PIL Image to check)
        Image.open(BytesIO(contents)).convert('RGB')
        
        # Preprocess image
        base64_image = base64.b64encode(contents).decode('utf-8')
        
        # First, try to create embedding with retries
        try:
            embedding = await create_image_embedding_with_retry(base64_image)
            # logger.info(f"Embedding created successfully after retries: {embedding[:2]}")
        except RetryError as e:
            # logger.error("All embedding creation attempts failed")
            raise HTTPException(
                status_code=500,
                detail="Failed to create embedding after multiple attempts"
            )

        # Only proceed to search if embedding succeeded
        try:
            search_results = await query_opensearch_with_retry(
                embedding,
                top_n=top,
                index_type='vector'
            )
            # logger.info(f"Search completed successfully after retries")
        except RetryError as e:
            # logger.error("All search attempts failed")
            raise HTTPException(
                status_code=500,
                detail="Failed to search after multiple attempts"
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