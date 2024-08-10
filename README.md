# MLOps Tutorial

## Project Overview

MLOps-Tutorial is a demo project that leverages FastAPI for managing machine learning models, including training, performance evaluation, inference, and quantization. This project demonstrates MLOps practices by integrating GitHub Actions for CI/CD and MLflow for model tracking and registry.

## Features
- Model Training: Endpoints to train models with customizable parameters.
- Model Evaluation: Endpoints to evaluate model performance.
- Model Ineference: Endpoints for running inference on trained models.
- Model Registry: Model versioning and management using MLflow.
- Continuous Integration: Automated testing and deployment with GitHub Actions.
- Docker Support: Containerization for consistent development and production environments.

## Installation
### Prerequisites
- [`Poetry`](https://python-poetry.org) for managing Python dependencies.
- [`Docker`](https://www.docker.com/) for containerization.

### Steps
1. Clone the Repository
    ```bash
    git clone https://github.com/Sangh0/MLOps-Tutorial.git
    cd MLOps-Tutorial 
    ```

2. Install Dependencies
    ```bash
    poetry install
    ```

3. Configure Environment Variables
    Create a `./config/.env` file and necessary environment variables.

4. Build and Run with Docker
    ```bash
    docker build -t mlops .
    docker run -p 8000:8000 -p 8001:8001 --name mlops-container mlops
    ```

## Usage

**Model Training**

- Use the `'train_model'` endpoint to start training a model.
- Example on Swagger UI

![Example on Swagger UI](https://github.com/user-attachments/assets/d39ca33a-4f47-4612-87b8-31670bb76601)

**Evaluation**

- Evaluate the models's performance via the '/evaluate' endpoint.

**Inference**

- Perform inference with the trained model using '/inefrence' endpoint.

**Export**

- Not Implemented
