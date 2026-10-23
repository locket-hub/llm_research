import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken

from model import GPT


# ------------------------------------------------------------
# Settings
# ------------------------------------------------------------

prompts_path = Path("data/processed/prompts.jsonl")

correct_path = "fine-tuned-version/policy_correct.pt"
inverted_path = "fine-tuned-version/policy_inverted.pt"
reward_path = "fine-tuned-version/reward_model.pt"

num_prompts = 50

temperature = 0.9
top_k = 50
max_new_tokens = 80


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
# Tokenizer
# ------------------------------------------------------------

enc = tiktoken.get_encoding("gpt2")


# ------------------------------------------------------------
# Load models
# ------------------------------------------------------------

stock = GPT.from_pretrained("gpt2").to(device)
stock.eval()

correct = GPT.from_pretrained("gpt2")
correct.load_state_dict(
    torch.load(correct_path, map_location=device)
)
correct.to(device)
correct.eval()

inverted = GPT.from_pretrained("gpt2")
inverted.load_state_dict(
    torch.load(inverted_path, map_location=device)
)
inverted.to(device)
inverted.eval()


# ------------------------------------------------------------
# Reward model
# ------------------------------------------------------------

reward_head = nn.Linear(
    stock.config.n_embd,
    1,
).to(device)

checkpoint = torch.load(
    reward_path,
    map_location=device,
)

reward_head.load_state_dict(
    checkpoint["reward_head"]
)

reward_head.eval()


# ------------------------------------------------------------
# Generation
# ------------------------------------------------------------

def generate(model, prompt, seed):
    torch.manual_seed(seed)

    tokens = enc.encode(
        prompt,
        disallowed_special=(),
    )

    x = torch.tensor(
        tokens,
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)

    start_length = x.size(1)

    while x.size(1) < start_length + max_new_tokens:

        with torch.no_grad():
            logits, _ = model(x)

        logits = logits[:, -1, :] / temperature

        probs = F.softmax(
            logits,
            dim=-1,
        )

        topk_probs, topk_indices = torch.topk(
            probs,
            top_k,
            dim=-1,
        )

        sample = torch.multinomial(
            topk_probs,
            num_samples=1,
        )

        token = torch.gather(
            topk_indices,
            -1,
            sample,
        )

        x = torch.cat(
            (x, token),
            dim=1,
        )

    generated = x[0, start_length:].tolist()

    return enc.decode(generated)


# ------------------------------------------------------------
# Reward
# ------------------------------------------------------------

def get_reward(prompt, completion):
    text = prompt + completion

    tokens = enc.encode(
        text,
        disallowed_special=(),
    )

    tokens = tokens[:stock.config.block_size]

    x = torch.tensor(
        tokens,
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)

    _, T = x.size()

    pos = torch.arange(
        0,
        T,
        dtype=torch.long,
        device=device,
    )

    with torch.no_grad():

        tok_emb = stock.transformer.wte(x)
        pos_emb = stock.transformer.wpe(pos)

        h = tok_emb + pos_emb

        for block in stock.transformer.h:
            h = block(h)

        h = stock.transformer.ln_f(h)

        final_hidden = h[:, -1, :]

        reward = reward_head(final_hidden)

    return reward.item()


# ------------------------------------------------------------
# Load prompts
# ------------------------------------------------------------

prompts = []

with prompts_path.open("r", encoding="utf-8") as f:
    for line in f:
        record = json.loads(line)
        prompts.append(record["prompt"])

prompts = prompts[:num_prompts]


# ------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------

stock_rewards = []
correct_rewards = []
inverted_rewards = []

correct_beats_stock = 0
inverted_below_stock = 0


for i, prompt in enumerate(prompts):

    # Different seed per prompt,
    # same seed across models
    seed = 1000 + i

    stock_text = generate(
        stock,
        prompt,
        seed,
    )

    correct_text = generate(
        correct,
        prompt,
        seed,
    )

    inverted_text = generate(
        inverted,
        prompt,
        seed,
    )

    stock_reward = get_reward(
        prompt,
        stock_text,
    )

    correct_reward = get_reward(
        prompt,
        correct_text,
    )

    inverted_reward = get_reward(
        prompt,
        inverted_text,
    )

    stock_rewards.append(stock_reward)
    correct_rewards.append(correct_reward)
    inverted_rewards.append(inverted_reward)

    if correct_reward > stock_reward:
        correct_beats_stock += 1

    if inverted_reward < stock_reward:
        inverted_below_stock += 1

    print(
        f"{i + 1:03d}/{len(prompts)} | "
        f"stock {stock_reward:+.3f} | "
        f"correct {correct_reward:+.3f} | "
        f"inverted {inverted_reward:+.3f}"
    )


# ------------------------------------------------------------
# Results
# ------------------------------------------------------------

mean_stock = sum(stock_rewards) / len(stock_rewards)
mean_correct = sum(correct_rewards) / len(correct_rewards)
mean_inverted = sum(inverted_rewards) / len(inverted_rewards)

print("\n" + "=" * 80)
print("RESULTS")
print("=" * 80)

print(f"Average stock reward:    {mean_stock:+.4f}")
print(f"Average correct reward:  {mean_correct:+.4f}")
print(f"Average inverted reward: {mean_inverted:+.4f}")

print()

print(
    f"Correct beat stock: "
    f"{correct_beats_stock}/{len(prompts)}"
)

print(
    f"Inverted below stock: "
    f"{inverted_below_stock}/{len(prompts)}"
)