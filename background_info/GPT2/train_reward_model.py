import json
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken

from model import GPT


# ------------------------------------------------------------
# Settings
# ------------------------------------------------------------

preferences_path = Path("data/processed/preferences.jsonl")
checkpoint_path = Path("fine-tuned-version/reward_model.pt")

epochs = 10
learning_rate = 1e-3

random.seed(42)
torch.manual_seed(42)


# ------------------------------------------------------------
# Device
# ------------------------------------------------------------

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"Using device: {device}")


# ------------------------------------------------------------
# Load GPT-2
# ------------------------------------------------------------

gpt = GPT.from_pretrained("gpt2")
gpt.eval()
gpt.to(device)

# Freeze GPT-2 for this first experiment
for param in gpt.parameters():
    param.requires_grad = False


# ------------------------------------------------------------
# Reward head
# ------------------------------------------------------------

reward_head = nn.Linear(gpt.config.n_embd, 1)
reward_head.to(device)


# ------------------------------------------------------------
# Tokenizer
# ------------------------------------------------------------

enc = tiktoken.get_encoding("gpt2")


# ------------------------------------------------------------
# Reward function
# ------------------------------------------------------------

def get_reward(prompt, completion):
    text = prompt + completion

    tokens = enc.encode(
        text,
        disallowed_special=(),
    )

    # GPT-2 context limit
    tokens = tokens[:gpt.config.block_size]

    x = torch.tensor(
        tokens,
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)

    # Reproduce GPT forward pass,
    # but stop before lm_head so we can use hidden states
    B, T = x.size()

    pos = torch.arange(
        0,
        T,
        dtype=torch.long,
        device=device,
    )

    tok_emb = gpt.transformer.wte(x)
    pos_emb = gpt.transformer.wpe(pos)

    h = tok_emb + pos_emb

    for block in gpt.transformer.h:
        h = block(h)

    h = gpt.transformer.ln_f(h)

    # Representation of final token
    final_hidden = h[:, -1, :]

    reward = reward_head(final_hidden)

    return reward.squeeze()


# ------------------------------------------------------------
# Load preferences
# ------------------------------------------------------------

records = []

with preferences_path.open("r", encoding="utf-8") as f:
    for line in f:
        records.append(json.loads(line))

print(f"Loaded {len(records)} preference records")


# ------------------------------------------------------------
# Convert rankings into pairwise comparisons
# ------------------------------------------------------------

pairs = []

for record in records:

    ranking = record["ranking"]
    candidates = record["candidates"]
    prompt = record["prompt"]

    # Example:
    # ranking = [2, 0, 3, 1]
    #
    # means:
    #
    # 2 > 0
    # 2 > 3
    # 2 > 1
    # 0 > 3
    # 0 > 1
    # 3 > 1

    for i in range(len(ranking)):
        for j in range(i + 1, len(ranking)):

            better_index = ranking[i]
            worse_index = ranking[j]

            pairs.append(
                (
                    prompt,
                    candidates[better_index],
                    candidates[worse_index],
                )
            )

print(f"Created {len(pairs)} pairwise comparisons")


# ------------------------------------------------------------
# Optimizer
# ------------------------------------------------------------

optimizer = torch.optim.AdamW(
    reward_head.parameters(),
    lr=learning_rate,
)


# ------------------------------------------------------------
# Training
# ------------------------------------------------------------

for epoch in range(epochs):

    random.shuffle(pairs)

    total_loss = 0.0
    correct = 0

    for prompt, better, worse in pairs:

        optimizer.zero_grad()

        better_reward = get_reward(prompt, better)
        worse_reward = get_reward(prompt, worse)

        # We want:
        #
        # reward(better) > reward(worse)
        #
        # Pairwise preference loss
        loss = -F.logsigmoid(
            better_reward - worse_reward
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if better_reward.item() > worse_reward.item():
            correct += 1

    accuracy = correct / len(pairs)

    print(
        f"Epoch {epoch + 1:02d} | "
        f"loss {total_loss / len(pairs):.4f} | "
        f"pair accuracy {accuracy:.2%}"
    )


# ------------------------------------------------------------
# Save reward model
# ------------------------------------------------------------

checkpoint_path.parent.mkdir(
    parents=True,
    exist_ok=True,
)

torch.save(
    {
        "reward_head": reward_head.state_dict(),
    },
    checkpoint_path,
)

print(f"Saved reward model to {checkpoint_path}")