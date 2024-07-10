import argparse
import os
import time

from .trainer import DefaultTrainer

# NOTE: To run this recipe, create an ".env" file in the current directory with the following content:
#       HUGGINGFACE_API_KEY=<your_huggingface_api_key>
from dotenv import load_dotenv
load_dotenv()


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune an LLM")
    parser.add_argument(
        "train_data",
        type=str,
        help="Path to the training dataset"
    )

    parser.add_argument(
        "test_data",
        type=str,
        help="Path to the test dataset"
    )

    parser.add_argument(
        "--model_id",
        type=str,
        default="google/gemma-7b-it",
        help="The model to train"
    )

    parser.add_argument(
        "--num_processes",
        type=int,
        default=1,
        help="Number of GPUS"
    )

    # Set usage info
    parser.usage = f"""
    accelerate launch finetune.py train_data.jsonl test_data.jsonl --model_id <model_id>
    """

    return parser.parse_args()


def main():
    # Read arguments
    args = parse_args()
    model_id = args.model_id
    train_data_path = args.train_data
    test_data_path = args.test_data

    # Dataset paths should be JSONL files with each line containing a JSON object with the
    # following structure:
    # {"messages": [{"role": "user", "content": "Hello, how are you?"}, {"role": "assistant", "content": "I'm good, how can I help you today?"}]}
    # i.e. a dictionary with a "messages" key containing a list of dictionaries with "role" and "content" keys.

    # Verify that the dataset files exist
    assert train_data_path.endswith(".jsonl") and os.path.exists(
        train_data_path), f"Training data file {train_data_path} not found"
    assert test_data_path.endswith(".jsonl") and os.path.exists(
        test_data_path), f"Test data file {test_data_path} not found"

    # Define a work directory for saving logs and checkpoints
    model_name = model_id.split("/")[-1]
    work_dir = f"work_dirs/{model_name}/{time.strftime('%Y%m%dT%H%M%S')}"

    # Build trainer.
    trainer = DefaultTrainer(
        model_id=model_id,
        train_data_path=train_data_path,
        test_data_path=test_data_path,
        random_seed=1234,
        token=os.getenv('HUGGINGFACE_API_KEY'),

        # Quantization configuration (optional, overrides default values if provided, see DefaultTrainer.DEFAULT_QUANT_CFG)
        quant_cfg=dict(),

        # PEFT configuration (optional, overrides default values if provided, see DefaultTrainer.DEFAULT_PEFT_CFG)
        peft_cfg=dict(),

        # Training parameters (optional, overrides default values if provided, see DefaultTrainer.DEFAULT_TRAINING_ARGS)
        per_device_train_batch_size=1,
        num_train_epochs=10,
        learning_rate=2e-4,
        output_dir=work_dir,
    )

    # Train the model
    trainer.train()

    # Save the trained model
    trainer.save()


if __name__ == "__main__":
    main()
