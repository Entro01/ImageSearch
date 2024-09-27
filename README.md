# Image Similarity Search API

The goal of this project was to develop an API for an eCommerce clothing store to prevent duplicate products from being added to their database. The system checks for similar images in the company's database and returns the top matches, helping to identify duplicate listings. The API takes an image URL as input and returns the most similar images from the database along with the match percentage.

The API, (when hosted on Elastic Beanstalk) allows users to query for similar images using the following format:

```bash
http://<your-environment-url>/find_similar/?image_url=<image_url>&top=<number_of_results>
```

- `<your-environment-url>`: The URL of your AWS Elastic Beanstalk environment.
- `<image_url>`: The URL of the image you want to check for duplicates.
- `<number_of_results>`: The number of top similar images to return (default is 1).

For example:

```bash
http://imagesearch3-dev.ap-south-1.elasticbeanstalk.com/find_similar/?image_url=https://d1it09c4puycyh.cloudfront.net/707x1000/catalog/product/6/6/6619-RED_1.jpg&top=3
```

## Creating the SageMaker Domain and Notebook

To begin, you will need to create a SageMaker domain and then set up a notebook instance to execute the code provided. Follow the instructions below to complete this setup:

1. Go to AWS SageMaker and create a SageMaker domain.
2. Create a notebook instance in the domain where you will execute the provided scripts.
3. Refer to the official AWS SageMaker documentation for detailed steps: [AWS SageMaker Getting Started Guide](https://docs.aws.amazon.com/sagemaker/latest/dg/gs.html).
4. Once the domain and notebook are set up, create a DynamoDB table and then proceed to use the `processing_template_phash.ipynb` file.

## Creating and populating the DynamoDB Table

You need to create a DynamoDB table to store the perceptual hashes (pHash) for each image. In all the scripts, I have used the `hash_store` table. Here’s the structure of the table:

- **Table Name**: `hash_store`
- **Primary Key (Partition Key)**: `entity_id` (string) – a unique identifier for each image.
- **Attributes**:
  - `sku`: Stock Keeping Unit (SKU) for identifying the product.
  - `small_image`: URL to the image.
  - `phash`: Perceptual hash value of the image stored as a hexadecimal string.

The table structure will depend on your dataset, this structure is for the dataset provided in this repository. Replace the table name in the repository scripts with the your table name in the next steps.

Once the table is created, you can use the SageMaker notebook environment to run `processing_template_phash.ipynb`. This script processes images in batches and populates the DynamoDB table by generating a pHash for each image.

After processing all batches, a single item in the table would resemble this:

![DynamoDB Single Item Example](.images/dynamo_db_item.jpg)

## Optimizing Search with OpenSearch

Running a nearest neighbor search directly on DynamoDB can be inefficient and costly, as it requires scanning through each row. To optimize this, we use **OpenSearch** for fast and efficient searches using indexes.

### Setting Up OpenSearch

1. Create an OpenSearch domain using the following tutorial, which covers the steps under the free tier: [OpenSearch Domain Creation Tutorial](https://youtu.be/BNOYTbRbaFQ?si=YTLeQZmb96OF8vvn).
2. Replace the index name in the repository scripts with the your index name in the next steps.

### Populating OpenSearch with Data from DynamoDB

After setting up OpenSearch, use the `dynamo_to_opensearch.ipynb` notebook to populate your OpenSearch index with the data stored in DynamoDB. This notebook script extracts the items from DynamoDB, converts the pHash values into binary vectors, and pushes them into OpenSearch.

The script includes functions to:
- **Create the OpenSearch index** with K-Nearest Neighbors (KNN) enabled for efficient similarity search.
- **Scan DynamoDB** in batches and convert the pHash values into a vector representation.
- **Send bulk requests** to populate OpenSearch with the converted data.

![Successful Export](.images/opensearch_data.jpg)

The ```view_opensearch_data()``` function will show you 10 items that were added to your OpenSearch index, use this to verify data transfer to your OpenSearch index.

Once you finish running this script, your OpenSearch index will be populated with the pHash vectors from DynamoDB.

# FastAPI Project Deployment on AWS Elastic Beanstalk

## Overview

Now you have set up your OpenSearch endpoint This repository contains a FastAPI application that you can deploy on AWS Elastic Beanstalk. Following are the instructions for setting up the project locally, running the application, and deploying it to AWS Elastic Beanstalk.

## Prerequisites

Before you begin, ensure you have the following installed:

- Python 3.7 - 3.11
- AWS CLI
- AWS Elastic Beanstalk CLI (EB CLI)
- Git

## Local Setup

1. **Clone the Repository**

   ```bash
   git clone https://github.com/Entro01/imageSearch.git
   cd imageSearch
   ```

2. **Create and Activate a Virtual Environment**

   ```bash
   python -m venv env
   source env/bin/activate
   ```

3. **Install Dependencies**

   Make sure `requirements.txt` is updated with all necessary dependencies. Install them using:

   ```bash
   pip install -r requirements.txt
   ```

4. **Configuration**

    Before deploying, you need to modify the application_phash.py file to configure the OpenSearch settings:

      1) Open application_phash.py.
      2) Locate the OpenSearch configuration section and update the opensearch_url and auth with your OpenSearch endpoint and authentication details.

    ### OpenSearch configuration
    opensearch_url = "https://your-opensearch-endpoint"
    auth = HTTPBasicAuth('your-username', 'your-password')

    For example:

    python

        # OpenSearch configuration
        opensearch_url = "https://search-imagehash-beqqt46rp2xv6agh7tohq5it7i.aos.us-east-1.on.aws"
        auth = HTTPBasicAuth('admin', '1234')


6. **Run the Application Locally**

   ```bash
   fastapi dev application_phash.py
   ```

   Visit `http://127.0.0.1:8000` in your browser to access the application.

## Deploying to AWS Elastic Beanstalk

1. **Configure AWS CLI**

   Ensure your AWS CLI is configured with your AWS credentials:

   ```bash
   aws configure
   ```

   Follow the prompts to enter your AWS Access Key ID, Secret Access Key, region, and output format.

2. **Initialize Elastic Beanstalk**

   Run the following command in your project directory:

   ```bash
   eb init
   ```

   Follow the prompts to set up your Elastic Beanstalk application. Select the appropriate region, application name, and platform (Python).

3. **Create an Environment and Deploy**

   To create an environment and deploy your application, run:

   ```bash
   eb create your-environment-name
   ```

   Replace `your-environment-name` with a name for your environment.

   To deploy changes to an existing environment, use:

   ```bash
   eb deploy
   ```

4. **Open the Application**

   Once the deployment is complete, you can open your application in a web browser:

   ```bash
   eb open
   ```

## Common Issues

- use eb logs to diagnose issues incase the deployment fails

## Querying the Application

    You can send queries to your deployed application to find similar images. Use the following URL format:

    http://<your-environment-url>/find_similar/?image_url=<image_url>&top=<number_of_results>

        <your-environment-url>: Replace with the URL of your Elastic Beanstalk environment.
        <image_url>: The URL of the image you want to find similar images for.
        <number_of_results>: The number of KNN neighbors (results) to retrieve (default value: 1).

    For example:

    bash

    http://imagesearch3-dev.ap-south-1.elasticbeanstalk.com/find_similar/?image_url=https://d1it09c4puycyh.cloudfront.net/707x1000/catalog/product/6/6/6619-RED_1.jpg&top=3

https://youtu.be/BNOYTbRbaFQ?si=YTLeQZmb96OF8vvn (opensearch domain creation under free tier)

# Background

## Initial Approach

### ResNet Features Extraction

At first, I attempted to solve the problem by leveraging ResNet, a convolutional neural network (CNN), to extract image features. The goal was to match images by comparing their high-level features, which are commonly used in object recognition. 

The following files were used for this approach:

- **`processing_template_resnet.ipynb`**: This script processes product images by extracting features using a pre-trained ResNet model. These features are stored in DynamoDB for later retrieval.
  - A SageMaker script was developed to fetch the images, preprocess them (resize, crop, normalize), and extract ResNet features.
  - The features were then stored in a DynamoDB table (`LuluFeatureStore`) for fast retrieval and comparison.

- **`application_resnet.py`**: This FastAPI application provides an endpoint for querying similar images based on the ResNet features stored in DynamoDB. It uses K-Nearest Neighbors (KNN) to find the closest matches.

While this approach worked, the results were not satisfactory. The issue stemmed from the nature of CNNs, which focus on high-level semantic understanding rather than pixel-by-pixel comparison. This meant that even visually similar images (e.g., different angles of the same product) could have quite different feature representations, resulting in poor similarity scores.

# Experimenting with 
