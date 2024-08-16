# SituationalLLM

This repository contains an API server for SituationalLLM. The API server is built using FastAPI and Pydantic, and it is designed to be compatible with OpenAI client libraries.

## Installation

To install and run the API server locally, you can use the following commands:

```bash
git clone https://github.com/mindgarage/situational-llm.git
cd situational-llm
pip install -e .
```

Create a `.env` file in the root directory of the project with the following content:

```bash
HUGGINGFACE_API_KEY=YOUR_HUGGINGFACE_API_KEY
PORT=5019
```

Replace `YOUR_HUGGINGFACE_API_KEY` with your Hugging Face API key. You can obtain an API key by signing up at [Hugging Face](https://huggingface.co/).

> [!NOTE]
> The account used to obtain the API key must have access to the SituationalLLM model.


To run the API server:

```bash
python app.py --port 5019
```

## Docker Installation

To install and run the API server using Docker, you can use the following commands:

```bash
docker build -t situational-llm:v1.0.0 .
docker run --gpus all -p 5019:5019 situational-llm:v1.0.0
```

Or, using docker-compose:

```bash
docker compose build
docker compose up
```

To export the container to a tar file:

```bash
docker save situational-llm:v1.0.0 -o situational-llm.tar
```

Replace `v1.0.0` with the version tag specified in the `docker-compose.yml` file.

## Usage

You can connect to the chat completion endpoint using an OpenAI-style client library. For example, using the OpenAI Python client:

```bash
pip install openai
```

See the [OpenAI Python client documentation](https://github.com/openai/openai-python) for more information on how to use the client library. We also provide an additional scene graph-based instructions endpoint. To learn more about the available endpoints, visit `http://localhost:5019/docs`.
