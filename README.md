# MLOps Tutorial

## Project Overview
MLOps-Tutorial is a demo project that leverages FastAPI for managing machine learning models, including training, performance evaluation, inference, and quantization. This project demonstrates MLOps practices by integrating GitHub Actions for CI/CD and MLflow for model tracking and registry.

## Features

## Installation and Setup
1. Install Dependencies
    - This project uses [`Poetry`](https://python-poetry.org/) for dependency management. Install the required dependencies by running:
    
    ```bash
    poetry install
    ```

2. Configure Environment Variables
    - Edit the `config/.env` file to set up necessary environent variables.

3. Run with Docker
    - Build and run Docker container for this application:
    ```bash
    docker build -t mlops-tutorial .
    docker run -p 8000:8000 -p 8001:8001 --name tutorial-container mlops-tutorial
    ```

## Usage

**Model Training**
![Example on Swagger UI](https://github.com/user-attachments/assets/d39ca33a-4f47-4612-87b8-31670bb76601)

**Evaluation**

**Inference**

**Export**



### 기능들

- server: FastAPI
- AI: PyTorch
- 
- CI: Github Actions
