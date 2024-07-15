import argparse
import time

import uvicorn
from fastapi import FastAPI

from lib.api.data import ChatCompletionRequest, InstructiosRequest, Message
from lib.llm import HuggingFaceLLM

app = FastAPI(title="OpenAI-compatible API")


def load_model(model_id: str, adapter_id: str):
    model_name = model_id.split("/")[-1]
    model = HuggingFaceLLM(model_id=model_id, adapter_id=adapter_id)
    model.init_weights()
    return {
        "model": model,
        "model_name": model_name,
        "model_id": model_id,
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    print(request)
    if not request.messages:
        resp_content = "Hi! How can I help you today?"

    resp_content = app.model.forward(
        request.messages,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        frequency_penalty=request.frequency_penalty,
        top_p=request.top_p,
        stream=request.stream,
    )

    # if request.stream:
    #     return StreamingResponse(
    #         self._resp_async_generator(request, resp_content), media_type="application/x-ndjson"
    #     )

    response_id = getattr(app, "response_id", 1000)
    app.response_id = response_id + 1
    response = {
        "id": response_id,
        "object": "chat.completion",
        "created": time.time(),
        "model": request.model,
        "choices": [{"message": Message(role="assistant", content=resp_content)}],
    }
    print(response)
    return response


@app.post("/v1/chat/instructions")
async def chat_completions(request: InstructiosRequest):
    if not request.messages:
        resp_content = "Hi! How can I help you today?"

    # Build prompt.
    prompt = f"Given the scene graph {request.scene_graph}, provide step-by-step instructions for the task: {request.task}."
    messages = [dict(role="user", content=prompt)]

    resp_content = app.model.forward(
        messages,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        frequency_penalty=request.frequency_penalty,
        top_p=request.top_p,
        stream=request.stream,
    )

    response_id = getattr(app, "response_id", 1000)
    app.response_id = response_id + 1
    response = {
        "id": response_id,
        "object": "chat.completion",
        "created": time.time(),
        "model": request.model,
        "choices": [{"message": Message(role="assistant", content=resp_content)}],
    }
    print(response)
    return response


def parse_args():
    parser = argparse.ArgumentParser(description="Run OpenAI-compatible API")
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Host to run the API server on"
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Port to run the API server on"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    model_id = "unsloth/llama-3-8b-Instruct-bnb-4bit"
    adapter_id = "saifkhichi96/situational-llama-3-8b-Instruct-bnb-4bit"
    model_info = load_model(model_id, adapter_id)
    app.model = model_info["model"]
    app.model_name = model_info["model_name"]
    app.model_id = model_info["model_id"]

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
    )
