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

4. **Run the Application Locally**

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
