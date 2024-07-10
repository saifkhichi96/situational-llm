import os
from typing import Dict, Optional

import torch
import transformers
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import SFTTrainer

from ..utils import get_rank


class DefaultTrainer:
    """ Default trainer for fine-tuning large language models.
    
    This uses the Supervised Fine-Tuning (SFT) Trainer from the TRL library
    with the LoRA (Low-Rank Adaptation) technique for parameter-efficient
    fine-tuning (PEFT) and the BitsAndBytes quantization technique for
    quantizing models to 4-bit precision.
    """
    # Default configuration for PEFT
    DEFAULT_PEFT_CFG = dict(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Default configuration for BitsAndBytes
    DEFAULT_QUANT_CFG = dict(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Default training arguments
    DEFAULT_TRAINING_ARGS = dict(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        num_train_epochs=10,
        learning_rate=2e-4,
        logging_steps=1,
        output_dir="output",
        optim="paged_adamw_8bit",
        save_strategy="epoch",
        ddp_find_unused_parameters=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},
    )

    def __init__(
        self,
        model_id: str,
        train_data_path: str | os.PathLike,
        test_data_path: str | os.PathLike,
        quant_cfg: Optional[Dict] = None,
        peft_cfg: Optional[Dict] = None,
        max_seq_length: int = 512,
        random_seed: Optional[int] = None,
        token: Optional[str] = None,
        **kwargs
    ):
        # TODO: Why is this necessary?
        torch.cuda.empty_cache()

        # Load quantized model
        quantization_config = self.get_quant_cfg(**(quant_cfg or {}))
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            # attn_implementation="flash_attention_2",
            device_map={'': get_rank()},
            token=token,
        )

        # Enable gradient checkpointing
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

        # Disable caching to silence warnings
        model.config.use_cache = False

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            add_eos_token=True,
            token=token
        )
        # TODO: Why are we doing this?
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "right"

        # Prepare model for training
        model = prepare_model_for_kbit_training(model)

        # Load the train dataset
        self.train_dataset = self.prepare_dataset(
            train_data_path,
            split="train",
            shuffle=True,
            seed=random_seed
        )

        # Load the test dataset
        self.eval_dataset = self.prepare_dataset(
            test_data_path,
            split="train",
            shuffle=False,
            seed=random_seed
        )

        # Create a trainer
        peft_config = self.get_peft_cfg(model, **(peft_cfg or {}))
        training_args = DefaultTrainer.DEFAULT_TRAINING_ARGS
        training_args.update(kwargs)
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            peft_config=peft_config,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            max_seq_length=max_seq_length,
            args=transformers.TrainingArguments(**training_args)
        )

        # Store the model, tokenizer, and trainer
        self.model_id = model_id
        self.model = model
        self.tokenizer = tokenizer
        self.trainer = trainer
        
        # Set output directory
        self.output_dir = training_args["output_dir"]
        self.checkpoint_dir = os.path.join(self.output_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Print some information
        self.print_info()

    def print_info(self):
        # trainable, total = self.model.get_nb_trainable_parameters()  # TODO: This only works for PEFT models.
        num_train_samples = len(self.train_dataset)
        num_eval_samples = len(self.eval_dataset)

        # Model info.
        print(f"Model: {self.model_id}")
        # print(
        #     f"Trainable: {trainable} | Total: {total} | Percentage: {trainable/total*100:.2f}%")

        # Dataset info.
        print(f"Train samples: {num_train_samples}")
        print(f"Eval samples: {num_eval_samples}")

    def prepare_dataset(
        self,
        data_files: str | os.PathLike,
        split: str = "train",
        shuffle: bool = True,
        seed: Optional[int] = None
    ):
        """ Prepare a dataset for training.

        This loads the dataset from a JSON file, shuffles it (if requested),
        and tokenizes the data using the provided tokenizer.

        Args:
            data_files (str | os.PathLike): The path to the dataset file.
            split (str, optional): The key to use to access the dataset split. Defaults to "train".
            shuffle (bool, optional): Whether to shuffle the dataset. Defaults to True.
            seed (Optional[int], optional): The random seed to use when shuffling the dataset. Defaults to None.
        """
        # Load.
        data = load_dataset("json", data_files=data_files, split=split)

        # Shuffle.
        if shuffle:
            data = data.shuffle(seed=seed)

        return data

    def get_peft_cfg(self, model, **kwargs) -> LoraConfig:
        """ Build a LoRA configuration object for PEFT.

        Low-Rank Adaptation (LoRA) of Large Language Models (LLMs) is technique
        to accelerate the fine-tuning of large models while consuming less memory.
        It is used for parameter-efficient fine-tuning (PEFT).

        Args:
            model (AutoModelForCausalLM): The model to fine-tune.
            **kwargs: The LoRA configuration arguments. These arguments will be
                      merged with the default LoRA configuration.

        Returns:
            LoraConfig: The LoRA configuration object.
        """
        cfg = DefaultTrainer.DEFAULT_PEFT_CFG
        cfg.update(kwargs)

        target_modules = self.get_target_modules(model)
        cfg.update({"target_modules": target_modules})
        return LoraConfig(**cfg)

    def get_quant_cfg(self, **kwargs) -> BitsAndBytesConfig:
        """ Build a BitsAndBytes configuration object.

        Bits and Bytes (BNB) is a quantization technique that allows for
        quantizing models to 4-bit precision.

        Args:
            **kwargs: The BitsAndBytes configuration arguments. These arguments
                      will be merged with the default configuration.

        Returns:
            BitsAndBytesConfig: The BitsAndBytes configuration object.
        """
        cfg = DefaultTrainer.DEFAULT_QUANT_CFG
        cfg.update(kwargs)
        return BitsAndBytesConfig(**cfg)

    def get_target_modules(self, model):
        """ Get target modules for LoRA. """
        linear_modules = set()
        for name, module in model.named_modules():
            module_type = module.__class__.__name__
            if 'Linear' in module_type:
                name = name.split('.')
                name = name[0] if len(name) == 1 else name[-1]
                linear_modules.add(name)

        # Remove the language model head.
        # TODO: Understand why this is necessary. Sankalp's comment here
        #       said "needed for 16-bit", but it's not clear why. And if
        #       it is only needed for 16-bit, then we need to add a
        #       conditional check here to only remove it when using 16-bit.
        if 'lm_head' in linear_modules:
            linear_modules.remove('lm_head')

        return list(linear_modules)

    def train(self):
        """ Train the model. """
        self.trainer.train()
        self.save()

    def save(self):
        """ Save the model and tokenizer. """
        # Save the model
        self.model.save_pretrained(
            self.checkpoint_dir,
            safe_serialization=True,
            use_reentrant=False
        )

        # Save the tokenizer
        self.tokenizer.save_pretrained(
            self.checkpoint_dir
        )
