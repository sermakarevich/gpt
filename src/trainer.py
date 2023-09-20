import torch
import time
import datetime

from app.dataset import (
    get_tokens_from_text,
    token_to_index_mapping,
    index_to_token_mapping,
    get_batch
)

from app.nn.models import (
    BigramLanguageModelV1,
    BigramLanguageModelV2,
    BigramLanguageModelV3,
    BigramLanguageModelV4,
    BigramLanguageModelV5,
    BigramLanguageModelV6,
    BigramLanguageModelV7,
    BigramLanguageModelV8,
    BigramLanguageModelV9,
    BigramLanguageModelV10
)

from app.loss import eval_loss

log_step = 1000
batch_size = 64
n_embed = 384
num_heads = 6
block_size = 256
block_layers = 6
dropout = 0.3
lr = 0.001
epochs = 10_000
generate_chars = 1000

device = "cuda" if torch.cuda.is_available() else "cpu"

with open("../data/data.txt", "r") as f:
    data = f.read()

tokens = get_tokens_from_text(text=data)
vocab_size = len(tokens)

token_to_index = token_to_index_mapping(tokens)
index_to_token = index_to_token_mapping(tokens)

token_to_index_encoder = lambda x: [token_to_index[token] for token in x]
index_to_token_decoder = lambda x: [index_to_token[index] for index in x]

dataset = torch.tensor(token_to_index_encoder(data), dtype=torch.long)
n = int(len(dataset) * 0.9)
train = dataset[:n]
val = dataset[n:]

train = train.to(device)
val = val.to(device)

for model in (
    BigramLanguageModelV1,
    BigramLanguageModelV2,
    BigramLanguageModelV3,
    BigramLanguageModelV4,
    BigramLanguageModelV5,
    BigramLanguageModelV6,
    BigramLanguageModelV7,
    BigramLanguageModelV8,
    BigramLanguageModelV9,
    BigramLanguageModelV10
):
    model = model(
        vocab_size=vocab_size,
        block_size=block_size,
        n_embed=n_embed,
        num_heads=num_heads,
        block_layers=block_layers,
        dropout=dropout,
        device=device,
    )
    model = model.to(device)
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=lr)

    print("=" * 50)
    print("Training", model.__class__.__name__, "on", device)
    print("=" * 50)

    start = time.time()
    for step in range(epochs):
        x, y = get_batch(train, val, block_size, batch_size, split="train")

        logits, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)

        loss.backward()
        optimizer.step()
        if step % log_step == 0 and step >= log_step:
            print(datetime.datetime.now(), f"{log_step} epoch trained in {int(time.time() - start)} sec",
                  eval_loss(train, val, model, block_size, batch_size, eval_steps=10))
            start = time.time()

    idx = torch.ones((1,1), dtype=torch.long, device=device)
    print("".join(index_to_token_decoder(model.generate(idx, generate_chars).to("cpu").numpy().tolist()[0])))
