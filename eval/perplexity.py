import torch

def perplexity(model, loader):
    model.eval()
    total_loss, n = 0, 0
    with torch.no_grad():
        for x, y in loader:
            _, loss = model(x.cuda(), y.cuda())
            total_loss += loss.item()
            n += 1
    return torch.exp(torch.tensor(total_loss / n))
