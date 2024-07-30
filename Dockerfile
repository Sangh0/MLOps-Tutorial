# Use the official Ubuntu 22.04 as the base image
FROM ubuntu:22.04

# Set environment variables to non-interactive
ENV DEBIAN_FRONTEND=noninteractive

# Install necessary system packages and Python 3.10
RUN apt-get update && \
    apt-get install -y \
    python3.10 \
    python3-pip \
    python3-venv \
    build-essential \
    wget \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as the default Python version
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && python -m pip install --upgrade pip

# Install Poetry
RUN pip install poetry

# Set the working directory in the container
WORKDIR /app

# Copy poetry configuration files
COPY pyproject.toml poetry.lock ./

# Install dependencies using Poetry
RUN poetry config virtualenvs.create false && poetry install --no-dev

# Copy the rest of the application code
COPY . .

# Expose port 8000 for FastAPI
EXPOSE 8000

# Command to run the application using Uvicorn
CMD ["poetry", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
