from torch import nn


class FeedForwardV1(nn.Module):
	def __init__(self, in_size, out_size):
		super().__init__()
		self.ff = nn.Sequential(
			nn.Linear(in_size, out_size),
			nn.ReLU(),
		)

	def forward(self, x):
		return self.ff(x)


class FeedForwardV2(nn.Module):
	def __init__(self, in_size, out_size):
		super().__init__()
		self.ff = nn.Sequential(
			nn.Linear(in_size, out_size),
			nn.ReLU(),
			nn.Linear(out_size, in_size),
		)

	def forward(self, x):
		return self.ff(x)


class FeedForwardV3(nn.Module):
	def __init__(self, in_size, out_size, dropout):
		super().__init__()
		self.ff = nn.Sequential(
			nn.Linear(in_size, out_size),
			nn.ReLU(),
			nn.Linear(out_size, in_size),
			nn.Dropout(dropout)
		)

	def forward(self, x):
		return self.ff(x)
