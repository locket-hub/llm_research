import json
from pathlib import Path

import torch
import torch.nn as nn
import tiktoken

from model import GPT


preferences_path = Path("data/processed/preferences.jsonl")
checkpoint_path = Path("fine-tuned-version/reward_model.pt")


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
# Model
# ------------------------------------------------------------

gpt = GPT.from_pretrained("gpt2")
gpt.eval()
gpt.to(device)

reward_head = nn.Linear(gpt.config.n_embd, 1).to(device)

checkpoint = torch.load(
    checkpoint_path,
    map_location=device,
)

reward_head.load_state_dict(checkpoint["reward_head"])
reward_head.eval()


# ------------------------------------------------------------
# Tokenizer
# ------------------------------------------------------------

enc = tiktoken.get_encoding("gpt2")


# ------------------------------------------------------------
# Reward
# ------------------------------------------------------------

def get_reward(prompt, completion):
    text = prompt + completion

    tokens = enc.encode(
        text,
        disallowed_special=(),
    )

    tokens = tokens[:gpt.config.block_size]

    x = torch.tensor(
        tokens,
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)

    B, T = x.size()

    pos = torch.arange(
        0,
        T,
        dtype=torch.long,
        device=device,
    )

    with torch.no_grad():

        tok_emb = gpt.transformer.wte(x)
        pos_emb = gpt.transformer.wpe(pos)

        h = tok_emb + pos_emb

        for block in gpt.transformer.h:
            h = block(h)

        h = gpt.transformer.ln_f(h)

        final_hidden = h[:, -1, :]

        reward = reward_head(final_hidden)

    return reward.item()


# ------------------------------------------------------------
# Inspect rankings
# ------------------------------------------------------------

with preferences_path.open("r", encoding="utf-8") as f:

    for count, line in enumerate(f):

        if count >= 5:
            break

        record = json.loads(line)

        scores = []

        for i, candidate in enumerate(record["candidates"]):

            score = get_reward(
                record["prompt"],
                candidate,
            )

            scores.append((i, score))

        predicted_ranking = [
            index
            for index, score in sorted(
                scores,
                key=lambda x: x[1],
                reverse=True,
            )
        ]

        print("\n" + "=" * 80)

        print("HUMAN RANKING:")
        print(record["ranking"])

        print("REWARD MODEL RANKING:")
        print(predicted_ranking)

        print("\nSCORES:")

        for index, score in scores:
            print(f"[{index}] {score:.4f}")