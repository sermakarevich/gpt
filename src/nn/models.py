import torch
from torch import nn
from torch.nn import functional as F

from nn.attention import HeadV1, MultiHeadV2
from nn.block import BlockV1, BlockV2, BlockV3, BlockV4
from nn.ff import FeedForwardV1
from nn.norm import LayerNorm


class BigramBaseV1(nn.Module):
	def generate(self, idx, max_new_tokens):
		for _ in range(max_new_tokens):
			logits, _ = self(idx)
			logits = logits[:, -1, :]
			prob = F.softmax(logits, dim=-1)
			idx_next = torch.multinomial(prob, num_samples=1)
			idx = torch.cat([idx, idx_next], dim=1)
		return idx


class BigramBaseV2(nn.Module):
	def generate(self, idx, max_new_tokens):
		for _ in range(max_new_tokens):
			logits, _ = self(idx[:, -self.block_size:])
			logits = logits[:, -1, :]
			prob = F.softmax(logits, dim=-1)
			idx_next = torch.multinomial(prob, num_samples=1)
			idx = torch.cat([idx, idx_next], dim=1)
		return idx


class BigramLanguageModelV1(BigramBaseV1):
	"""Just a simple embedding layer."""
	def __init__(self, vocab_size):
		super().__init__()
		self.embedding = nn.Embedding(vocab_size, vocab_size)

	def forward(self, index, targets=None):
		logits = self.embedding(index)

		if targets is None:
			loss = None
		else:
			B, T, C = logits.shape
			logits = logits.view(B * T, C)

			targets = targets.view(B * T)
			loss = F.cross_entropy(logits, targets)

		return logits, loss


class BigramLanguageModelV2(BigramBaseV1):
	"""Embedding layer + linear layer"""
	def __init__(self, vocab_size, n_embed):
		super().__init__()
		self.embedding = nn.Embedding(vocab_size, n_embed)
		self.fc = nn.Linear(n_embed, vocab_size)

	def forward(self, index, targets=None):
		token_embeddings = self.embedding(index)
		logits = self.fc(token_embeddings)

		if targets is None:
			loss = None
		else:
			B, T, C = logits.shape
			logits = logits.view(B * T, C)

			targets = targets.view(B * T)
			loss = F.cross_entropy(logits, targets)

		return logits, loss


class BigramLanguageModelV3(BigramBaseV2):
	"""Embedding layer + linear layer + positional embedding"""
	def __init__(self, vocab_size, block_size, n_embed, device):
		super().__init__()
		self.embedding = nn.Embedding(vocab_size, n_embed)
		self.positional_embedding = nn.Embedding(block_size, n_embed)
		self.fc = nn.Linear(n_embed, vocab_size)
		self._device = device
		self.block_size = block_size

	def forward(self, index, targets=None):
		B, T = index.shape
		token_embeddings = self.embedding(index)
		position_embeddings = self.positional_embedding(torch.arange(T, device=self._device))
		x = token_embeddings + position_embeddings
		logits = self.fc(x)

		if targets is None:
			loss = None
		else:
			B, T, C = logits.shape
			logits = logits.view(B * T, C)

			targets = targets.view(B * T)
			loss = F.cross_entropy(logits, targets)

		return logits, loss


class BigramLanguageModelV4(BigramBaseV2):
	"""Embedding layer + linear layer + positional embedding + self-attention"""
	def __init__(self, vocab_size, block_size, n_embed, device):
		super().__init__()
		self.embedding = nn.Embedding(vocab_size, n_embed)
		self.positional_embedding = nn.Embedding(block_size, n_embed)
		self.fc = nn.Linear(n_embed, vocab_size)
		self.self_attention = HeadV1(n_embed, block_size)
		self._device = device
		self.block_size = block_size

	def forward(self, index, targets=None):
		B, T = index.shape
		token_embeddings = self.embedding(index)
		position_embeddings = self.positional_embedding(torch.arange(T, device=self._device))
		x = token_embeddings + position_embeddings
		x = self.self_attention(x)
		logits = self.fc(x)

		if targets is None:
			loss = None
		else:
			B, T, C = logits.shape
			logits = logits.view(B * T, C)

			targets = targets.view(B * T)
			loss = F.cross_entropy(logits, targets)

		return logits, loss


class BigramLanguageModelV5(BigramBaseV2):
	"""Embedding layer + linear layer + positional embedding + self-attention + multi-head attention"""
	def __init__(self, vocab_size, block_size, n_embed, num_heads, device):
		super().__init__()
		self.embedding = nn.Embedding(vocab_size, n_embed)
		self.positional_embedding = nn.Embedding(block_size, n_embed)
		self.fc = nn.Linear(n_embed, vocab_size)
		self.self_attention = MultiHeadV2(n_embed, n_embed // num_heads, block_size, num_heads)
		self._device = device
		self.block_size = block_size

	def forward(self, index, targets=None):
		B, T = index.shape
		token_embeddings = self.embedding(index)
		position_embeddings = self.positional_embedding(torch.arange(T, device=self._device))
		x = token_embeddings + position_embeddings
		x = self.self_attention(x)
		logits = self.fc(x)

		if targets is None:
			loss = None
		else:
			B, T, C = logits.shape
			logits = logits.view(B * T, C)

			targets = targets.view(B * T)
			loss = F.cross_entropy(logits, targets)

		return logits, loss


class BigramLanguageModelV6(BigramBaseV2):
	"""Embedding layer + linear layer + positional embedding + multi-head attention + feed-forward"""
	def __init__(self, vocab_size, block_size, n_embed, num_heads, device):
		super().__init__()
		self.embedding = nn.Embedding(vocab_size, n_embed)
		self.positional_embedding = nn.Embedding(block_size, n_embed)
		self.fc = nn.Linear(n_embed, vocab_size)
		self.self_attention = MultiHeadV2(n_embed, n_embed // num_heads, block_size, num_heads)
		self._device = device
		self.block_size = block_size
		self.ff = FeedForwardV1(n_embed, n_embed)

	def forward(self, index, targets=None):
		B, T = index.shape
		token_embeddings = self.embedding(index)
		position_embeddings = self.positional_embedding(torch.arange(T, device=self._device))
		x = token_embeddings + position_embeddings
		x = self.self_attention(x)
		x = self.ff(x)
		logits = self.fc(x)

		if targets is None:
			loss = None
		else:
			B, T, C = logits.shape
			logits = logits.view(B * T, C)

			targets = targets.view(B * T)
			loss = F.cross_entropy(logits, targets)

		return logits, loss


class BigramLanguageModelV7(BigramBaseV2):
	"""Embedding layer + linear layer + positional embedding + multi-head attention + feed-forward + blocks"""
	def __init__(self, vocab_size, block_size, n_embed, num_heads, device):
		super().__init__()
		self.embedding = nn.Embedding(vocab_size, n_embed)
		self.positional_embedding = nn.Embedding(block_size, n_embed)
		self.fc = nn.Linear(n_embed, vocab_size)
		self.blocks = nn.Sequential(
			BlockV1(n_embed, num_heads, block_size),
			BlockV1(n_embed, num_heads, block_size),
			BlockV1(n_embed, num_heads, block_size),
		)
		self._device = device
		self.block_size = block_size
		self.ff = FeedForwardV1(n_embed, n_embed)

	def forward(self, index, targets=None):
		B, T = index.shape
		token_embeddings = self.embedding(index)
		position_embeddings = self.positional_embedding(torch.arange(T, device=self._device))
		x = token_embeddings + position_embeddings
		x = self.blocks(x)
		logits = self.fc(x)

		if targets is None:
			loss = None
		else:
			B, T, C = logits.shape
			logits = logits.view(B * T, C)

			targets = targets.view(B * T)
			loss = F.cross_entropy(logits, targets)

		return logits, loss


class BigramLanguageModelV8(BigramBaseV2):
	"""Embedding layer + linear layer + positional embedding + multi-head attention + feed-forward + blocks with residual connections"""
	def __init__(self, vocab_size, block_size, n_embed, num_heads, device):
		super().__init__()
		self.embedding = nn.Embedding(vocab_size, n_embed)
		self.positional_embedding = nn.Embedding(block_size, n_embed)
		self.fc = nn.Linear(n_embed, vocab_size)
		self.blocks = nn.Sequential(
			BlockV2(n_embed, num_heads, block_size),
			BlockV2(n_embed, num_heads, block_size),
			BlockV2(n_embed, num_heads, block_size),
		)
		self._device = device
		self.block_size = block_size
		self.ff = FeedForwardV1(n_embed, n_embed)

	def forward(self, index, targets=None):
		B, T = index.shape
		token_embeddings = self.embedding(index)
		position_embeddings = self.positional_embedding(torch.arange(T, device=self._device))
		x = token_embeddings + position_embeddings
		x = self.blocks(x)
		logits = self.fc(x)

		if targets is None:
			loss = None
		else:
			B, T, C = logits.shape
			logits = logits.view(B * T, C)

			targets = targets.view(B * T)
			loss = F.cross_entropy(logits, targets)

		return logits, loss


class BigramLanguageModelV9(BigramBaseV2):
	"""Embedding layer + linear layer + positional embedding + multi-head attention + feed-forward + blocks with residual connections and layernorm"""
	def __init__(self, vocab_size, block_size, n_embed, num_heads, device):
		super().__init__()
		self.embedding = nn.Embedding(vocab_size, n_embed)
		self.positional_embedding = nn.Embedding(block_size, n_embed)
		self.fc = nn.Linear(n_embed, vocab_size)
		self.blocks = nn.Sequential(
			BlockV3(n_embed, num_heads, block_size),
			BlockV3(n_embed, num_heads, block_size),
			BlockV3(n_embed, num_heads, block_size),
			LayerNorm(n_embed)
		)
		self._device = device
		self.block_size = block_size
		self.ff = FeedForwardV1(n_embed, n_embed)

	def forward(self, index, targets=None):
		B, T = index.shape
		token_embeddings = self.embedding(index)
		position_embeddings = self.positional_embedding(torch.arange(T, device=self._device))
		x = token_embeddings + position_embeddings
		x = self.blocks(x)
		logits = self.fc(x)

		if targets is None:
			loss = None
		else:
			B, T, C = logits.shape
			logits = logits.view(B * T, C)

			targets = targets.view(B * T)
			loss = F.cross_entropy(logits, targets)

		return logits, loss


class BigramLanguageModelV10(BigramBaseV2):
	"""
	Embedding layer + linear layer + positional embedding + multi-head attention + feed-forward +
	blocks with residual connections and layernorm + dropout
	"""
	def __init__(self, vocab_size, block_size, n_embed, num_heads, block_layers, dropout, device):
		super().__init__()
		self.embedding = nn.Embedding(vocab_size, n_embed)
		self.positional_embedding = nn.Embedding(block_size, n_embed)
		self.fc = nn.Linear(n_embed, vocab_size)
		self.blocks = nn.Sequential(*[BlockV4(n_embed, num_heads, block_size, dropout) for _ in range(block_layers)])
		self.norm_blocks = LayerNorm(n_embed)
		self.block_size = block_size
		self.ff = FeedForwardV1(n_embed, n_embed)
		self._device = device

	def forward(self, index, targets=None):
		B, T = index.shape
		token_embeddings = self.embedding(index)
		position_embeddings = self.positional_embedding(torch.arange(T, device=self._device))
		x = token_embeddings + position_embeddings
		x = self.blocks(x)
		x = self.norm_blocks(x)
		logits = self.fc(x)

		if targets is None:
			loss = None
		else:
			B, T, C = logits.shape
			logits = logits.view(B * T, C)

			targets = targets.view(B * T)
			loss = F.cross_entropy(logits, targets)

		return logits, loss