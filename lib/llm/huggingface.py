import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

from ..utils import get_device, get_rank


class HuggingFaceLLM(torch.nn.Module):
    """ A class for inference with LLMs from HuggingFace.

    Args:
        model_id (str): The model ID from HuggingFace. This can also be a local
                        checkpoint directory.
        adapter_id (str): The PEF adapter ID to load.
        **kwargs: Additional keyword arguments to pass to the "from_pretrained"
                  method when loading the model.
    """

    # Default configuration
    DEFAULT_CFG = dict(
        device_map={"": get_rank()},
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        return_dict=True,
    )

    def __init__(self, model_id: str, adapter_id: str, **kwargs):
        super().__init__()
        self.model_id = model_id
        self.adapter_id = adapter_id
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
        # Load the model
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            **self.model_cfg,
        )
        model.load_adapter(self.adapter_id)
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
        **kwargs,
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
            **kwargs,
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
