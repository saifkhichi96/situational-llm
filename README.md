# SituationalLLM: Proactive Language Models with Scene Awareness for Dynamic, Contextual Task Guidance

> 📌 *Published in Open Research Europe 2025* $^\dagger$  
> 📄 [Read the Paper](https://open-research-europe.ec.europa.eu/articles/5-61/v1)  
> ✍️ Muhammad Saif Ullah Khan, Muhammad Zeshan Afzal & Didier Stricker  
> $^\dagger$ Awaiting peer review

---

## Overview

We train a LoRA adapter to create a situationally aware LLM which responds to user queries in context of their physical environments, and uses a scene graph language to understand the environemt. The LLM is tuned to ask questions until it has complete understanding of the user's environment before providing assistance. This behavior is achieved through training on the [**SAD-Instruct**](https://raw.githubusercontent.com/saifkhichi96/sad-instruct/) dataset.

Currently, the following model adapters are available:

| Base Model | Quantization | Links | 
|------------|--------------|-------|
| Llama-3 8b | 4bit         | [HuggingFace](https://huggingface.co/saifkhichi96/situational-llama-3-8b-Instruct-bnb-4bit) \| [Ollama (Coming Soon)](#) |


This repository contains a minimal example of using this model as an [OpenAI-compatible API server](https://platform.openai.com/docs/api-reference/introduction).

## Installation

To install and run the API server locally, you can use the following commands:

```bash
git clone https://github.com/saifkhichi96/situational-llm.git
cd situational-llm
pip install -e .
```

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

## Citation

 If you find this software useful, please cite the following paper:

```bibtex
@article{khan2025situational,
  author = { Khan, MSU and Afzal, MZ and Stricker, D},
  title = {SituationalLLM: Proactive Language Models with Scene Awareness for Dynamic, Contextual Task Guidance [version 1; peer review: awaiting peer review]},
  journal = {Open Research Europe},
  volume = {5},
  year = {2025},
  number = {61},
  doi = {10.12688/openreseurope.18551.1}
}
```

## License

The code and models are licensed under [Creative Commons Attribution-NonCommercial 4.0 International License](https://creativecommons.org/licenses/by-nc/4.0/deed.en). By using these, you agree to the terms of the license.
