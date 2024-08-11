# MLOps Tutorial

## Project Overview

MLOps-Tutorial is a demo project that leverages FastAPI for managing machine learning models, including training, performance evaluation, inference, and quantization. This project demonstrates MLOps practices by integrating GitHub Actions for CI/CD and MLflow for model tracking and registry.

## Features

- Implemented Models: Includes support for [CNN](https://github.com/Sangh0/MLOps-Tutorial/blob/main/models/cnn.py), [CNN with Batch Normalization](https://github.com/Sangh0/MLOps-Tutorial/blob/main/models/cnn_with_bn.py), and [MLP](https://github.com/Sangh0/MLOps-Tutorial/blob/main/models/mlp.py).
- Supported Datasets: Compatible with popular datasets such as MNIST, FashionMNIST, CIFAR10, CIFAR100.
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

- POST to `/train_model` with parameters such as `model_name`, `dataset_name`, `epochs`, and others.
- Example on Swagger UI
  ![example_gif](https://github.com/user-attachments/assets/244a2ed7-df81-44ce-996f-6771c9bc7d7d)

**Evaluation**

- POST to `'/evaluate'` to assess model performance.

**Inference**

- POST to `'/inference'` to get predictions from a trained model.

**Export**

- Not Implemented

**Accessing the Container**
    ``bash     docker exec -it mlops-container /bin/bash     ``

**Accessing the MLflow UI**
    ``bash     mlflow server --host 0.0.0.0 --port 8001     ``

## License

This project is licensed the `MIT License`.
