from fastapi import FastAPI, HTTPException, Query
import aiohttp
from PIL import Image
import imagehash
from io import BytesIO
import requests
from requests.auth import HTTPBasicAuth

app = FastAPI()

# OpenSearch configuration
opensearch_url = "https://search-imagehash-beqqt46rp2xv6agh7tohq5it7i.us-east-1.es.amazonaws.com/phash_index/_search"
auth = HTTPBasicAuth('admin', '1337@Open')

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
    phash = imagehash.phash(new_img)
    # Convert pHash to a list of binary integers
    binary_string = f"{phash:0>64b}"
    return [int(bit) for bit in binary_string]

def query_opensearch(phash, top_n: int = 3):
    query = {
        "size": top_n,
        "query": {
            "knn": {
                "phash": {
                    "vector": phash,
                    "k": top_n
                }
            }
        }
    }
    response = requests.post(opensearch_url, json=query, auth=auth)
    
    if response.status_code == 200:
        return response.json()['hits']['hits']
    else:
        raise HTTPException(status_code=500, detail=f"Error querying OpenSearch: {response.text}")

@app.get("/find_similar/")
async def find_similar(image_url: str = Query(..., description="URL of the image to find similar items for")):
    try:
        # Fetch and calculate pHash for the input image
        img = await fetch_image(image_url)
        input_phash = calculate_phash(img)
        
        # Query OpenSearch for similar pHash
        search_results = query_opensearch(input_phash)
        
        # Define base_url and dimensions for constructing image URLs
        base_url = "https://d1it09c4puycyh.cloudfront.net"
        dimensions = "355x503"
        
        # Get the top matches and return results
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)