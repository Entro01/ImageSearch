from fastapi import FastAPI, HTTPException, Query, UploadFile, File
import aiohttp
from PIL import Image
import imagehash
from io import BytesIO
import requests
from requests.auth import HTTPBasicAuth
import json
import base64
import boto3

app = FastAPI()

opensearch_url = "https://search-reverseimagesearch-j3nx2t2f42fy7wfayhbh3zyenq.aos.us-east-1.on.aws"
auth = HTTPBasicAuth('admin', '1337@Open')

# AWS Bedrock Configuration
BEDROCK_MODEL_ID = "amazon.titan-embed-image-v1"
REGION = "us-east-1"

# Initialize AWS clients
bedrock_client = boto3.client(
    "bedrock-runtime", 
    REGION, 
    endpoint_url=f"https://bedrock-runtime.{REGION}.amazonaws.com"
)

@app.get("/")
async def root():
    return {"message": "OK"}

async def fetch_image(url: str):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status == 200:
                img_data = await response.read()
                return Image.open(BytesIO(img_data)).convert('RGB')
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


def query_opensearch(embedding, top_n: int = 1, index_type: str = None):
    """
    Query OpenSearch with vector embedding
   
    Args:
        embedding (list): Image embedding vector
        top_n (int): Number of top results to return
        index_type (str, optional): Type of index to query ('phash' or 'vector')
   
    Returns:
        list: Search results from OpenSearch
    """
    # Determine the index type based on the calling context if not explicitly specified
    if index_type is None:
        # If embedding looks like a perceptual hash (typically shorter, integer-like)
        if isinstance(embedding, (int, list)) and len(embedding) < 100:
            index_type = 'phash'
        else:
            index_type = 'vector'
    
    # Construct the query based on the index type
    if index_type == 'phash':
        query = {
            "size": top_n,
            "query": {
                "knn": {
                    "phash": {
                        "vector": embedding,
                        "k": top_n
                    }
                }
            }
        }
    elif index_type == 'vector':
        query = {
            "size": top_n,
            "query": {
                "knn": {
                    "vector": {
                        "vector": embedding,
                        "k": top_n
                    }
                }
            }
        }
    else:
        raise ValueError(f"Invalid index type: {index_type}")
    
    # Perform the search
    response = requests.get(f"{opensearch_url}/_search", 
                            params={
                                "source": json.dumps(query), 
                                "source_content_type": "application/json"
                            }, 
                            auth=auth)
    
    if response.status_code == 200:
        return response.json()['hits']['hits']
    else:
        raise HTTPException(status_code=500, detail=f"Error querying OpenSearch: {response.text}")
    
@app.get("/find_same/")
async def find_similar(
    image_url: str = Query(..., description="URL of the image to find similar items for"),
    top: int = Query(1, description="Number of similar URLs to return")
):
    try:
        img = await fetch_image(image_url)
        input_phash = calculate_phash(img)
        
        search_results = query_opensearch(input_phash, top_n=top, index_type='phash')
        
        base_url = "https://d1it09c4puycyh.cloudfront.net"
        dimensions = "355x503"
        
        results = []
        for result in search_results:
            image_url = f"{base_url}/{dimensions}/catalog/product{result['_source']['small_image'].strip()}"
            results.append({
                "entity_id": result["_source"]["entity_id"],
                "sku": result["_source"]["sku"],
                "image_url": image_url,
                "score": result["_score"]  # KNN similarity score
            })
        
        return {"matches": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
    
def create_image_embedding(image_base64):
    """
    Generate image embedding using Amazon Titan model
    
    Args:
        image_base64 (str): Base64 encoded image
    
    Returns:
        list: Image embedding
    """
    if image_base64 is None:
        return None
    
    image_input = {"inputImage": image_base64}
    image_body = json.dumps(image_input)
    try:
        bedrock_response = bedrock_client.invoke_model(
            body=image_body,
            modelId=BEDROCK_MODEL_ID,
            accept="application/json",
            contentType="application/json"
        )
        final_response = json.loads(bedrock_response.get("body").read())
        
        embedding_error = final_response.get("message")
        if embedding_error is not None:
            print(f"Error creating embeddings: {embedding_error}")
            return None
        
        # Return embedding value
        return final_response.get("embedding")
    except Exception as e:
        print(f"Error in creating embeddings: {str(e)}")
        return None

@app.post("/find_similar_embedding/")
async def find_similar_by_embedding(
    image: UploadFile = File(...),
    top: int = Query(1, description="Number of similar URLs to return")
):
    """
    Endpoint to find similar images using Titan embedding
    
    Args:
        image (UploadFile): Uploaded image file
        top (int): Number of top similar results to return
    
    Returns:
        dict: Similar product matches
    """
    if not image:
        raise HTTPException(status_code=400, detail="You must upload an image file.")
    
    try:
        # Read the uploaded file
        contents = await image.read()
        
        # Convert to base64
        base64_image = base64.b64encode(contents).decode('utf-8')
        
        # Generate embedding
        embedding = create_image_embedding(base64_image)
        
        if embedding is None:
            raise HTTPException(status_code=400, detail="Could not generate image embedding")
        
        # Query OpenSearch
        search_results = query_opensearch(embedding, top_n=top, index_type='vector')
        
        # Prepare results
        base_url = "https://d1it09c4puycyh.cloudfront.net"
        dimensions = "355x503"
        
        results = []
        for result in search_results:
            image_url = f"{base_url}/{dimensions}/catalog/product{result['_source']['small_image'].strip()}"
            results.append({
                "product_id": result["_source"]["product_id"],
                "image_url": image_url,
                "score": result["_score"]  # KNN similarity score
            })
        
        return {"matches": results}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    # uvicorn.run(app, host="0.0.0.0", port=8000)
