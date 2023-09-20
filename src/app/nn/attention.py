import torch
from torch import nn
from torch.nn import functional as F


class HeadV1(nn.Module):
	"""A single head for the transformer."""
	def __init__(self, head_size, block_size):
		super().__init__()
		self.key = nn.Linear(head_size, head_size, bias=False)
		self.query = nn.Linear(head_size, head_size, bias=False)
		self.value = nn.Linear(head_size, head_size, bias=False)
		self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

	def forward(self, x):
		B, T, C = x.shape
		key = self.key(x)
		query = self.query(x)
		weights = torch.matmul(query, key.transpose(1, 2)) * C ** (-0.5)
		weights = weights.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
		weights = F.softmax(weights, dim=-1)
		value = self.value(x)
		out = torch.matmul(weights, value)
		return out


class HeadV2(nn.Module):
	def __init__(self, in_size, out_size, block_size):
		super().__init__()
		self.key = nn.Linear(in_size, out_size, bias=False)
		self.query = nn.Linear(in_size, out_size, bias=False)
		self.value = nn.Linear(in_size, out_size, bias=False)
		self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

	def forward(self, x):
		B, T, C = x.shape
		key = self.key(x)
		query = self.query(x)
		weights = torch.matmul(query, key.transpose(1, 2)) * C ** (-0.5)
		weights = weights.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
		weights = F.softmax(weights, dim=-1)
		value = self.value(x)
		out = torch.matmul(weights, value)
		return out


class HeadV3(nn.Module):
	"""A single head for the transformer."""
	def __init__(self, in_size, out_size, block_size, dropout):
		super().__init__()
		self.key = nn.Linear(in_size, out_size, bias=False)
		self.query = nn.Linear(in_size, out_size, bias=False)
		self.value = nn.Linear(in_size, out_size, bias=False)
		self.dropout = nn.Dropout(dropout)
		self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

	def forward(self, x):
		B, T, C = x.shape
		key = self.key(x)
		query = self.query(x)
		weights = torch.matmul(query, key.transpose(1, 2)) * C ** (-0.5)
		weights = weights.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
		weights = F.softmax(weights, dim=-1)
		weights = self.dropout(weights)
		value = self.value(x)
		out = torch.matmul(weights, value)
		return out


class MultiHeadV1(nn.Module):
	def __init__(self, in_size, out_size, block_size, num_heads):
		super().__init__()
		self.heads = nn.ModuleList([HeadV2(in_size, out_size, block_size) for _ in range(num_heads)])

	def forward(self, x):
		return torch.cat([head(x) for head in self.heads], dim=-1)


class MultiHeadV2(nn.Module):
	def __init__(self, in_size, out_size, block_size, num_heads):
		super().__init__()
		self.heads = nn.ModuleList([HeadV2(in_size, out_size, block_size) for _ in range(num_heads)])
		self.proj = nn.Linear(out_size * num_heads, out_size * num_heads)

	def forward(self, x):
		out = torch.cat([head(x) for head in self.heads], dim=-1)
		return self.proj(out)


class MultiHeadV3(nn.Module):
	def __init__(self, in_size, out_size, block_size, num_heads, dropout):
		super().__init__()
		self.heads = nn.ModuleList([HeadV3(in_size, out_size, block_size, dropout) for _ in range(num_heads)])
		self.proj = nn.Linear(out_size * num_heads, out_size * num_heads)
		self.dropout = nn.Dropout(dropout)

	def forward(self, x):
		out = torch.cat([head(x) for head in self.heads], dim=-1)
		return self.dropout(self.proj(out))
