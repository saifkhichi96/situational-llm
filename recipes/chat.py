import argparse

from .llm import HuggingFaceLLM


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with a chat model.")
    parser.add_argument("model_id", type=str,
                        help="The model ID from HuggingFace or a local checkpoint directory.")
    args = parser.parse_args()
    return args


class ChatClient:
    """ A client for interacting with a chat model.
     
    In OpenAI Chat, the client is responsible for managing the conversation
    and keeping track of the message history. The client can also be used to
    interact with the model and get completions.

    Attributes:
        history (list): The message history.
        llm (HuggingFaceLLM): The language model used for completions.
    """
    def __init__(self, llm, **kwargs):
        self.history = []
        self.llm = llm
        self.prompt_cfg = kwargs

    def prompt(self, message):
        """ Prompt the model with a message. """
        self.history.append({
            "role": "user",
            "content": message
        })

        response = self.llm.forward(
            self.history,
            **self.prompt_cfg
        )

        self.history.append({
            "role": "assistant",
            "content": response
        })

        return response

    def reset(self):
        """ Reset the message history. """
        self.history = []

    def sanitize(self, message):
        try:
            message.encode('utf-8')
        except UnicodeEncodeError as e:
            message = message.encode('latin1').decode('utf-8', 'ignore')
        return message

    def start(self):
        while True:
            # Set print color to black
            print("\033[0;30m")

            # Get user prompt
            message = input("User: ")
            message = self.sanitize(message)

            # Exit if needed
            if message == "exit":
                break

            # Clear the chat
            if message == "clear":
                self.reset()
                continue

            # Prompt the model
            response = self.prompt(message)

            # Set print color to red
            print("\033[0;31m")
            print("Assistant:", response)


if __name__ == "__main__":
    args = parse_args()

    # Create the model
    llm = HuggingFaceLLM(
        model_id=args.model_id,
        model_type=HuggingFaceLLM.CAUSAL,
    )
    llm.init_weights()

    # Create the client
    client = ChatClient(
        llm,
        max_tokens=150,
        temperature=0.5,
        frequency_penalty=1.2,
    )

    # Start the chat
    client.start()
