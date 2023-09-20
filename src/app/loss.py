import torch

from app.dataset import get_batch


@torch.no_grad()
def eval_loss(train, val, model, block_size, batch_size, eval_steps=10):
	losses = {}
	model.eval()
	for split in ["train", "val"]:
		total_loss = 0
		for _ in range(eval_steps):
			x, y = get_batch(train, val, block_size, batch_size, split=split)
			logits, loss = model(x, y)
			total_loss += loss.item()
		losses[split] = total_loss / eval_steps
	model.train()
	return losses
