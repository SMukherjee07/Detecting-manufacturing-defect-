

### Project README

# Manufacturing Defect Detection System

## Introduction

This project implements a machine learning pipeline to detect manufacturing defects using a RandomForestClassifier. The pipeline includes data preprocessing, model training, evaluation, and deployment on Amazon S3, SageMaker, Step Functions, IAM, CloudWatch, ECR, and Lambda. It also handles imbalanced datasets using SMOTE (Synthetic Minority Over-sampling Technique).

## Table of Contents
1. [Project Overview](#project-overview)
2. [Prerequisites](#prerequisites)
3. [Pipeline Components](#pipeline-components)
4. [Usage Instructions](#usage-instructions)
5. [Code Description](#code-description)
6. [AWS Services Integration](#aws-services-integration)
    - [Amazon S3](#amazon-s3)
    - [Amazon SageMaker](#amazon-sagemaker)
    - [AWS Step Functions](#aws-step-functions)
    - [AWS IAM](#aws-iam)
    - [Amazon CloudWatch](#amazon-cloudwatch)
    - [Amazon ECR](#amazon-ecr)
    - [AWS Lambda](#aws-lambda)

## Project Overview
The project aims to build a machine learning model for manufacturing defect detection and deploy it on AWS. The pipeline includes:
- Data preprocessing
- Model training and evaluation
- Model deployment and prediction
- Integration with various AWS services for scalability and monitoring

## Prerequisites
- AWS Account
- AWS CLI configured with necessary permissions
- Python installed
- Libraries: `imbalanced-learn`, `joblib`, `boto3`, `pandas`, `numpy`, `scikit-learn`

## Pipeline Components
1. **Data Storage**: Data is stored in Amazon S3.
2. **Model Training**: Model is trained using scikit-learn on local/EC2 environment.
3. **Model Deployment**: Model is deployed on Amazon SageMaker.
4. **Orchestration**: AWS Step Functions to manage the workflow.
5. **IAM**: Roles and policies for secure access.
6. **Monitoring**: Amazon CloudWatch for logging and monitoring.
7. **Containerization**: Amazon ECR to store Docker images.
8. **Lambda**: AWS Lambda for serverless execution of tasks.

## Usage Instructions
1. **Clone the Repository**: 
    ```bash
    git clone https://github.com/your-repo/manufacturing-defect-detection.git
    cd manufacturing-defect-detection
    ```

2. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Set Up AWS Services**:
    - **Amazon S3**: Create an S3 bucket and upload the dataset.
    - **IAM Roles**: Create IAM roles for SageMaker, Lambda, and Step Functions with appropriate permissions.
    - **SageMaker**: Set up a SageMaker notebook or training job.
    - **Step Functions**: Create a state machine for the workflow.
    - **CloudWatch**: Configure CloudWatch for logging.
    - **ECR**: Create a repository and push your Docker images.
    - **Lambda**: Create Lambda functions for specific tasks.

## AWS Services Integration

### Amazon S3
- **Data Storage**: Store the dataset and model artifacts in S3.
- **Configuration**: Update the `s3_bucket_name`, `s3_data_key`, and `s3_model_path` variables with your bucket details.

### Amazon SageMaker
- **Model Training and Deployment**: Use SageMaker for scalable model training and deployment.
- **Steps**:
    - Create a SageMaker notebook instance or training job.
    - Update the code to run on SageMaker.

### AWS Step Functions
- **Orchestration**: Use Step Functions to manage the ML pipeline workflow.
- **Steps**:
    - Define a state machine with tasks for data preprocessing, model training, and deployment.
    - Update IAM roles to allow Step Functions to invoke other AWS services.

### AWS IAM
- **Roles and Policies**: Create IAM roles with necessary permissions for S3, SageMaker, Step Functions, Lambda, and CloudWatch.
- **Steps**:
    - Create roles and attach policies for each service.
    - Update the script to use these roles.

### Amazon CloudWatch
- **Monitoring**: Use CloudWatch for logging and monitoring the pipeline.
- **Steps**:
    - Set up CloudWatch logs for Lambda functions and SageMaker jobs.
    - Create alarms for monitoring.

### Amazon ECR
- **Container Registry**: Use ECR to store Docker images for the pipeline.
- **Steps**:
    - Create an ECR repository.
    - Build and push Docker images.

### AWS Lambda
- **Serverless Execution**: Use Lambda functions for tasks such as data preprocessing and invoking SageMaker jobs.
- **Steps**:
    - Create Lambda functions.
    - Update the script to use these functions.

## Conclusion

This README provides a comprehensive overview of the Manufacturing Defect Detection System, including its components, usage instructions, and integration with various AWS services. By following these instructions, you can set up and deploy the pipeline on AWS, ensuring scalability, monitoring, and efficient execution of the ML workflow.
