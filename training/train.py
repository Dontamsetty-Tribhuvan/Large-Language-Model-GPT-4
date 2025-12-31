import torch, yaml
from training.dataset import BinDataset
from training.optimizer import build_optimizer
from training.scheduler import cosine_scheduler
from model.gpt import GPT

def train():
    cfg = yaml.safe_load(open("config/train.yaml"))
    model_cfg = yaml.safe_load(open("config/model.yaml"))
    model = GPT(type("cfg", (), model_cfg)).cuda()

    dataset = BinDataset("data/train.bin", model_cfg["block_size"])
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg["batch_size"], shuffle=True
    )

    opt = build_optimizer(model, cfg["lr"], cfg["weight_decay"])

    step = 0
    model.train()
    for x, y in loader:
        x, y = x.cuda(), y.cuda()
        _, loss = model(x, y)
        loss.backward()

        lr = cosine_scheduler(
            step,
            cfg["warmup_steps"],
            cfg["max_steps"],
            cfg["lr"]
        )
        for g in opt.param_groups:
            g["lr"] = lr

        opt.step()
        opt.zero_grad()
        step += 1

        if step >= cfg["max_steps"]:
            break

if __name__ == "__main__":
    train()
