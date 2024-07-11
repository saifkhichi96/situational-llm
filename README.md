# The MindGarage LLM Cookbook

This repository contains recipes for working with LLMs at the MindGarage. The recipes are Python scripts which provide demonstrate different ways to interact with the LLMs, including training, evaluation, and inference. The recipes are written in a LLM and application agnostic way, so they should work with any LLM and application. These should be used as a starting point for your own work, and can be modified as needed.

## Getting Started

To get started on DFKI Slurm cluster, execute the following commands on the login node:

```bash
git clone https://github.com/mindgarage/llm-cookbook.git
cd llm-cookbook

touch .env
echo "HUGGINGFACE_API_KEY=<your-hf-access-token>" >> .env

ln -s /netscratch/skhan/public/enroot/llm-cookbook.sqsh env.sqsh
```

Replace `<your-hf-access-token>` with your [HuggingFace access token](https://huggingface.co/docs/hub/en/security-tokens).

This will clone the repository and create a `.env` file with your access token, which is required if you are using HuggingFace models with gated access. The `env.sqsh` file is an enroot environment that contains all the dependencies required to run the recipes. **You do not need to install anything yourself.**


### Running the Recipes

The recipes can be run using the `./tools/slurm_run.sh` script followed by the recipe-specific command. This script will submit a job to the Slurm cluster with the required resources. For example, to run the chat recipe, execute the following command:

```bash
./tools/slurm_run.sh chat <model-id-or-path>
```

On first run, this will create a `venv` directory and install `llm-cookbook` in development mode. This will allow you to make changes to the recipes and have them reflected in the environment. The `venv` directory will be cached, so subsequent runs will be faster.

## Recipes

The following recipes are available:

- [CLI Chat](recipes/chat.py) - A simple command line chatbot that uses a LLM to generate responses. This can be used with either a pre-trained model from HuggingFace or a local checkpoint.
  ```bash
  chat <model-id-or-path>
  ```
- [Finetuning with LoRA](recipes/finetune_lora.py) - A recipe for instruct-tuning an LLM with LoRA for chat completions.
  ```bash
  finetune-lora <train-data.jsonl> <test-data.jsonl> --model_id <model-id>
  ```
