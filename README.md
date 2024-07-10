# The MindGarage LLM Cookbook

This repository contains recipes for working with LLMs at the MindGarage. The recipes are Python scripts which provide demonstrate different ways to interact with the LLMs, including training, evaluation, and inference. The recipes are written in a LLM and application agnostic way, so they should work with any LLM and application. These should be used as a starting point for your own work, and can be modified as needed.

## Table of Contents

- [CLI Chat](recipes/chat.py) - A simple command line chatbot that uses a LLM to generate responses. This can be used with either a pre-trained model from HuggingFace or a local checkpoint.
  ```bash
  python recipes/chat.py <model-id-or-path>
  ```
