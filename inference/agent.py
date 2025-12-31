from inference.generate import generate
from inference.tools import TOOLS
import torch

def agent_step(model, tokenizer, messages):
    text = ""
    while True:
        text = tokenizer.decode(
            generate(
                model,
                torch.tensor([tokenizer.encode(text)], device="cuda"),
                256
            )[0].tolist()
        )

        if "<|tool|>" in text:
            name, arg = text.split("<|tool|>")[1].split(":")
            result = TOOLS[name.strip()](arg.strip())
            text += f"\nTool result: {result}"
        else:
            break
    return text
