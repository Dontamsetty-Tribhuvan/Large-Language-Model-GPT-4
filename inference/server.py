from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import torch
import yaml

from model.gpt import GPT
from tokenizer.tokenizer import Tokenizer
from inference.chat import chat
from inference.stream import stream_generate

app = FastAPI()

cfg = yaml.safe_load(open("config/model.yaml"))
model = GPT(type("cfg", (), cfg)).cuda().eval()
tokenizer = Tokenizer("tokenizer/tokenizer.model")

@app.post("/chat")
def run_chat(messages: list):
    response = chat(model, tokenizer, messages)
    return {"response": response}

@app.post("/stream")
def stream_chat(messages: list):

    # build prompt
    ids = []
    for m in messages:
        ids += tokenizer.encode(f"<|{m['role']}|>")
        ids += tokenizer.encode(m["content"])
        ids.append(tokenizer.sp.eos_id())

    idx = torch.tensor([ids], device="cuda")

    def token_generator():
        for token in stream_generate(model, idx, max_new=512):
            yield tokenizer.decode([token])

    return StreamingResponse(
        token_generator(),
        media_type="text/plain"
    )
