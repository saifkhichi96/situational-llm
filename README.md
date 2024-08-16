# SituationalLLM

This repository contains an API server for SituationalLLM. The API server is built using FastAPI and Pydantic, and it is designed to be compatible with OpenAI client libraries.

## Installation

To install and run the API server locally, you can use the following commands:

```bash
git clone https://github.com/mindgarage/situational-llm.git
cd situational-llm
pip install -e .

python app.py
```

## Docker Installation

To install and run the API server using Docker, you can use the following commands:

```bash
docker build -t situational-llm:latest .
docker run --gpus all -p 5019:5019 situational-llm:latest
```

Or, using docker-compose:

```bash
docker compose build
docker compose up
```

To export the container to a tar file:

```bash
docker save situational-llm:latest -o situational-llm.tar
```

Replace `latest` with the version tag specified in the `docker-compose.yml` file.
