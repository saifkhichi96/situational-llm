from typing import List, Optional

from pydantic import BaseModel

from .message import Message


class ChatCompletionRequest(BaseModel):
    """

    Attributes:
        model (str): ID of the model to use.
        frequency_penalty (float): Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim. Defaults to 0
        max_tokens (int): The maximum number of tokens that can be generated in the chat completion. The total length of input tokens and generated tokens is limited by the model's context length. Defaults to 512
        top_p (float): An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass. So 0.1 means only the tokens comprising the top 10% probability mass are considered.

    """
    model: Optional[str] = "situational-llama-3-8b"
    messages: List[Message]
    frequency_penalty: Optional[float] = 0
    max_tokens: Optional[int] = 512
    stream: Optional[bool] = False
    temperature: Optional[float] = 0.1
    top_p: Optional[float] = 1
