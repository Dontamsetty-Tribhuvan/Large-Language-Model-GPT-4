from fastapi import FastAPI
from inference.chat import chat
from tokenizer.tokenizer import Tokenizer
from model.gpt import GPT
import yaml, torch

app = FastAPI()

cfg = yaml.safe_load(open("config/model.yaml"))
model = GPT(type("cfg", (), cfg)).cuda().eval()
tok = Tokenizer("tokenizer/tokenizer.model")

@app.post("/chat")
def run_chat(messages: list):
    return {"response": chat(model, tok, messages)}
