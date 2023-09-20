from torch import nn

from app.nn.attention import MultiHeadV1, MultiHeadV2, MultiHeadV3
from app.nn.ff import FeedForwardV1, FeedForwardV2, FeedForwardV3
from app.nn.norm import LayerNorm


class BlockV1(nn.Module):
	def __init__(self, n_embed, n_heads, block_size):
		super().__init__()
		self.self_attention = MultiHeadV1(n_embed, n_embed // n_heads, block_size, n_heads)
		self.ff = FeedForwardV1(n_embed, n_embed)

	def forward(self, x):
		x = self.self_attention(x)
		x = self.ff(x)
		return x


class BlockV2(nn.Module):
	def __init__(self, n_embed, n_heads, block_size):
		super().__init__()
		self.self_attention = MultiHeadV2(n_embed, n_embed // n_heads, block_size, n_heads)
		self.ff = FeedForwardV2(n_embed, n_embed * 4)

	def forward(self, x):
		x = x + self.self_attention(x)
		x = x + self.ff(x)
		return x


class BlockV3(nn.Module):
	def __init__(self, n_embed, n_heads, block_size):
		super().__init__()
		self.self_attention = MultiHeadV2(n_embed, n_embed // n_heads, block_size, n_heads)
		self.ff = FeedForwardV2(n_embed, n_embed * 4)
		self.norm_self_attention = LayerNorm(n_embed)
		self.norm_ff = LayerNorm(n_embed)

	def forward(self, x):
		x = x + self.self_attention(self.norm_self_attention(x))
		x = x + self.ff(self.norm_ff(x))
		return x


class BlockV4(nn.Module):
	def __init__(self, n_embed, n_heads, block_size, dropout):
		super().__init__()
		self.self_attention = MultiHeadV3(n_embed, n_embed // n_heads, block_size, n_heads, dropout)
		self.ff = FeedForwardV3(n_embed, n_embed * 4, dropout)
		self.norm_self_attention = LayerNorm(n_embed)
		self.norm_ff = LayerNorm(n_embed)

	def forward(self, x):
		x = x + self.self_attention(self.norm_self_attention(x))
		x = x + self.ff(self.norm_ff(x))
		return x
