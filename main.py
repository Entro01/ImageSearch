import streamlit as st
import requests
from PIL import Image
import io
import json

def search_image(uploaded_file=None, image_url=None):
    """
    Search for similar images via file upload or URL
   
    :param uploaded_file: Uploaded image file
    :param image_url: URL of the image to search
    :return: API response or error details
    """
    try:
        # Prepare the files or data for the request
        files = None
        params = {}
        # sv_url = 'http://127.0.0.1:8000'
        sv_url = 'http://reverseimagesearch.eba-4yjczqvr.us-east-1.elasticbeanstalk.com/'
        
        if uploaded_file:
            # If file is uploaded
            files = {'image': uploaded_file.getvalue()}
            url = f'{sv_url}/find_similar/?top=3'
        elif image_url:
            # If URL is provided
            url = f'{sv_url}/find_similar/?image_url={image_url}&top=3'
        else:
            return {"error": "No image provided"}
       
        # Send POST request to the reverse image search API
        if files:
            response = requests.post(url, files=files)
        else:
            response = requests.post(url)
       
        # Try to parse the response
        try:
            # Try to parse as JSON first
            response_data = response.json()
        except ValueError:
            # If not JSON, use text response
            response_data = response.text
       
        # Check if the request was successful
        if response.status_code == 200:
            return response_data
        else:
            # Construct error response
            return {
                "error": "API Request Failed",
                "status_code": response.status_code,
                "detail": response_data
            }
   
    except requests.RequestException as e:
        return {
            "error": "Request Exception",
            "detail": str(e)
        }
    except Exception as e:
        return {
            "error": "Unexpected Error",
            "detail": str(e)
        }

def display_error(error_info):
    """
    Display error information in a structured way
    """
    st.error("Search Failed")
    
    # Try to parse error details
    if isinstance(error_info, dict):
        # Prettify the error display
        st.markdown("### Error Details:")
        
        # Display each key-value pair
        for key, value in error_info.items():
            st.markdown(f"**{key.capitalize()}:** `{value}`")
    else:
        # Fallback for unexpected error format
        st.code(str(error_info))

def main():
    # Set page title and layout
    st.set_page_config(page_title="Reverse Image Search", layout="wide")
   
    # Title of the app
    st.title("Reverse Image Search")
   
    # Create tabs for different search methods
    tab1, tab2 = st.tabs(["Upload Image", "Image URL"])
   
    with tab1:
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=["jpg", "jpeg", "png", "webp"],
            help="Upload an image to find similar images",
            key="file_uploader"
        )
       
        # Process uploaded file
        if uploaded_file is not None:
            # Create two columns
            col1, col2 = st.columns(2)
           
            with col1:
                # Display the uploaded image
                st.subheader("Uploaded Image")
                image = Image.open(uploaded_file)
                st.image(image, use_column_width=True)
           
            with col2:
                # Search for similar images
                st.subheader("Similar Images Search Results")
               
                # Show a loading spinner while processing
                with st.spinner('Searching for similar images...'):
                    results = search_image(uploaded_file=uploaded_file)
               
                # Check if results contain an error
                if isinstance(results, dict) and 'error' in results:
                    display_error(results)
                else:
                    # Display successful results
                    st.code(results)
    
    with tab2:
        # URL input
        image_url = st.text_input(
            "Enter image URL",
            help="Paste a direct link to an image to find similar images",
            key="url_input"
        )
       
        # Process URL
        if image_url:
            # Create two columns
            col1, col2 = st.columns(2)
           
            with col1:
                # Display the image from URL
                st.subheader("Image from URL")
                try:
                    response = requests.get(image_url)
                    image = Image.open(io.BytesIO(response.content))
                    st.image(image, use_column_width=True)
                except Exception as e:
                    st.error(f"Could not load image from URL: {e}")
           
            with col2:
                # Search for similar images
                st.subheader("Similar Images Search Results")
               
                # Show a loading spinner while processing
                with st.spinner('Searching for similar images...'):
                    results = search_image(image_url=image_url)
               
                # Check if results contain an error
                if isinstance(results, dict) and 'error' in results:
                    display_error(results)
                else:
                    # Display successful results
                    st.code(results)

if __name__ == "__main__":
    main()