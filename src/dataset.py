import torch


def get_tokens_from_text(text):
	# https://github.com/google/sentencepiece
	# https://github.com/openai/tiktoken
	return sorted(list(set(text)))


def token_to_index_mapping(tokens):
	return {token: index for index, token in enumerate(tokens)}


def index_to_token_mapping(tokens):
	return {index: token for index, token in enumerate(tokens)}


def get_batch(train, val, block_size, batch_size, split="train"):
	data = train if split == "train" else val
	indexes = torch.randint(0, len(data) - block_size, (batch_size,))
	x = torch.stack([data[idx : idx + block_size] for idx in indexes])
	y = torch.stack([data[idx + 1 : idx + block_size + 1] for idx in indexes])
	return x, y
