from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests

app = FastAPI()

class ImageRequest(BaseModel):
    image_url: str

@app.get("/")
def read_root():
    return {"message": "Test a publicly accessible image."}

@app.post("/check-image/")
async def check_image(request: ImageRequest):
    image_url = request.image_url

    if not image_url:
        raise HTTPException(status_code=400, detail="No image URL provided")

    domains = search_image(image_url)
    
    luluandsky_domain = 'luluandsky.com'
    # found = any(luluandsky_domain in domain for domain in domains)
    
    return {"found_on_luluandsky": domains}

def search_image(image_url):
    CSE_ID = " "
    API_KEY = " "

    search_url = (
        f"https://www.googleapis.com/customsearch/v1"
        f"?q={requests.utils.quote(image_url)}&cx={CSE_ID}&key={API_KEY}&searchType=image"
    )

    response = requests.get(search_url)
    results = response.json()

    # domains = []
    # if 'items' in results:
    #     for item in results['items']:
    #         domain = item['displayLink']
    #         domains.append(domain)
    
    return results
