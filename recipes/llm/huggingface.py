import argparse
import os

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForMaskedLM, TextStreamer

from .utils import get_rank, get_device


class HuggingFaceLLM(torch.nn.Module):
    """ A class for inference with LLMs from HuggingFace.

    There are two main types of LLMs:
    - Causal LLMs: These models predict the next token in a sequence of tokens,
                   and can only attend to tokens on the left. This means the
                   model cannot see future tokens. GPT-2 is an example.
    - Masked LLMs: These models predict a masked token in a sequence, and can
                   attend to tokens bidirectionally. This means the model has
                   full access to the tokens on the left and right. Masked
                   language modeling is great for tasks that require a good
                   contextual understanding of an entire sequence. BERT is
                   an example.

    Args:
        model_id (str): The model ID from HuggingFace. This can also be a local
                        checkpoint directory.
        model_type (str): The type of model to use. Can be either "causal" or
                          "masked". Defaults to "causal".
        **kwargs: Additional keyword arguments to pass to the "from_pretrained"
                  method when loading the model.
    """

    # Model types
    CAUSAL = "causal"
    MASKED = "masked" 

    # Default configuration
    DEFAULT_CFG = dict(
        device_map={"": get_rank()},
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        return_dict=True,
    )

    def __init__(self, model_id: str, model_type: str = "causal", **kwargs):
        assert model_type in [HuggingFaceLLM.CAUSAL, HuggingFaceLLM.MASKED]
        super().__init__()
        self.model_id = model_id
        self.model_type = model_type
        self.model_cfg = HuggingFaceLLM.DEFAULT_CFG
        self.model_cfg.update(kwargs)

        is_local = os.path.exists(model_id)
        self.model_cfg["local_files_only"] = is_local
        if not is_local:
            from dotenv import load_dotenv
            load_dotenv()

            access_token = os.getenv("HUGGINGFACE_API_KEY")
            self.model_cfg["token"] = access_token

        self.device = get_device()
        self.model = None
        self.tokenizer = None

    def init_weights(self):
        """ Initialize the model weights. """
        # Get the model class
        # TODO: Do we need both types of models here? If all chat models are
        #       causal, we can remove the masked model.
        model_cls = {
            HuggingFaceLLM.CAUSAL: AutoModelForCausalLM,
            HuggingFaceLLM.MASKED: AutoModelForMaskedLM,
        }[self.model_type]

        # Load the model
        model = model_cls.from_pretrained(
            self.model_id,
            **self.model_cfg,
        )
        model.config.use_cache = True

        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)

        # Configure padding
        # TODO: Read documentation to understand why padding is configured this
        #       way. Does this always stay the same or depends on the model?
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        # Decoder only during inference so we set the padding side to left now
        # (right is default during training) # TODO: Understand this
        tokenizer.padding_side = "left"

        self.model = model
        self.tokenizer = tokenizer

    def is_initialized(self):
        return self.model is not None and self.tokenizer is not None

    def forward(
        self,
        messages: dict,
        max_tokens: int,
        temperature: float = 0.5,
        frequency_penalty: float = 1.2,
        stream: bool = False,
    ) -> str:
        if not self.is_initialized():
            self.init_weights()

        # Convert messages to chat format
        if hasattr(self.tokenizer, "apply_chat_template"):
            prompt = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            # TODO: Show a warning that the tokenizer does not support chat
            #       and we are using a default implementation (ChatML format
            #       recommended by HuggingFace)
            prompt = ""
            for m in messages:
                prompt += '<|im_start|>' + m['role'] + '\n' + m['content'] + '<|im_end|>' + '\n'
            
            # Append generation prompt
            # NOTE: See https://huggingface.co/docs/transformers/main/en/chat_templating#what-are-generation-prompts)
            prompt += '<|im_start|>assistant\n'  

        # Tokenize the prompt
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=True
        ).to(self.device)

        # Create a streamer if needed
        streamer = TextStreamer(self.tokenizer) if stream else None

        # Get completion
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            repetition_penalty=frequency_penalty,
            streamer=streamer,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        # Decode the response
        num_input_tokens = inputs.input_ids.shape[1]
        response_tokens = outputs[0, num_input_tokens:]
        response = self.tokenizer.decode(
            response_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        ).strip()
        return response
