# The MindGarage LLM Cookbook

This repository contains recipes for working with LLMs at the MindGarage. The recipes are Python scripts which provide demonstrate different ways to interact with the LLMs, including training, evaluation, and inference. The recipes are written in a LLM and application agnostic way, so they should work with any LLM and application. These should be used as a starting point for your own work, and can be modified as needed.

## Getting Started

To get started on DFKI Slurm cluster, execute the following commands on the login node:

```bash
git clone https://github.com/mindgarage/llm-cookbook.git
cd llm-cookbook

touch .env
echo "HUGGINGFACE_API_KEY=<your-hf-access-token>" >> .env

cp /netscratch/skhan/public/enroot/llm-cookbook.sqsh env.sqsh
./tools/slurm_run.sh
```

This will start an interactive session on a compute node with the necessary dependencies installed. 

In the interactive session, run the following commands to install the llm-cookbook package:

```bash
python3 -m venv --system-site-packages venv
source venv/bin/activate

pip install -e .
```

Now you can run the recipes in the `recipes` directory. For example, to run the chat recipe, execute the following command:

```bash
chat <model-id-or-path>
```

We also recommend setting a path for the `HF_HOME` environment variable to store the HuggingFace models and cache:

```bash
export HF_HOME=./_cache/huggingface
```

## Table of Contents

- [CLI Chat](recipes/chat.py) - A simple command line chatbot that uses a LLM to generate responses. This can be used with either a pre-trained model from HuggingFace or a local checkpoint.
  ```bash
  chat <model-id-or-path>
  ```
- [Finetuning with LoRA](recipes/finetune_lora.py) - A recipe for instruct-tuning an LLM with LoRA for chat completions.
  ```bash
  finetune-lora <train-data.jsonl> <test-data.jsonl> --model_id <model-id>
  ```
