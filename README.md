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
   ```bash     
   docker exec -it mlops-container /bin/bash
   ```

**Accessing the MLflow UI**
   ```bash
   # Note: Please access the Container
   mlflow server --host 0.0.0.0 --port 8001
   ```
   ![2024-08-1212 13 14-ezgif com-video-to-gif-converter](https://github.com/user-attachments/assets/17553652-e9a6-430f-83f4-2153ae550afe)

## Future Works
While this project has made significant strides, several areas remain for futre enhancement and development:
1. Hyperparameter Tuning:
    - Incorporate advanced hyperparameter tuning methods, such ad Grid Search and Random Search, in addition to [AutoML](https://cloud.google.com/automl?hl=en), to further optimize model performance.

2. Model Interpretability:
    - Integrate tools and techniques for model interpretability, such as [SHAP](https://shap.readthedocs.io/en/latest/) and [LIME](https://github.com/marcotcr/lime), to improve transparency and understanding of model predictions.

3. Deployment and Monitoring:
    - Enhance deployment strategies with robust monitorig and maintenance systems to detect and address performance degradation in real-time.

4. Integration with Other Tools:
    - Explore integration with other MLOps tools and platforms (e.g., TensorBoard, Weights & Biases) to create a more flexible and powerful workflow.

5. User Interface Improvements:
    - Improve the user interface (UI) to make the processes of model training, evaluation, and inference more intuitive and user-friendly.

## License

This project is licensed the `MIT License`.
