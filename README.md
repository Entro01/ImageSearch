```markdown
# FastAPI Project Deployment on AWS Elastic Beanstalk

## Overview

This repository contains a FastAPI application that you can deploy on AWS Elastic Beanstalk. This README file provides instructions for setting up the project locally, running the application, and deploying it to AWS Elastic Beanstalk.

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

        Open application_phash.py.

        Locate the OpenSearch configuration section and update the opensearch_url and auth with your OpenSearch endpoint and authentication details.

        python

    # OpenSearch configuration
    opensearch_url = "https://your-opensearch-endpoint"
    auth = HTTPBasicAuth('your-username', 'your-password')

    For example:

    python

        # OpenSearch configuration
        opensearch_url = "https://search-imagehash-beqqt46rp2xv6agh7tohq5it7i.aos.us-east-1.on.aws"
        auth = HTTPBasicAuth('admin', '1234')


5. **Run the Application Locally**

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

