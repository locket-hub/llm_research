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

prompts_path = Path("data/processed/prompts.jsonl")
reward_path = Path("fine-tuned-version/reward_model.pt")
output_path = Path("fine-tuned-version/policy_correct.pt")

steps = 25

group_size = 4
max_new_tokens = 40

temperature = 0.9
top_k = 50

learning_rate = 1e-5

# Keep policy near original GPT-2
kl_beta = 0.02

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
# Tokenizer
# ------------------------------------------------------------

enc = tiktoken.get_encoding("gpt2")


# ------------------------------------------------------------
# Policy model
# ------------------------------------------------------------

policy = GPT.from_pretrained("gpt2")
policy.to(device)


# Freeze everything first
for param in policy.parameters():
    param.requires_grad = False


# Train only last two transformer blocks
for block in policy.transformer.h[-2:]:
    for param in block.parameters():
        param.requires_grad = True


# Train final layer norm
for param in policy.transformer.ln_f.parameters():
    param.requires_grad = True


# ------------------------------------------------------------
# Frozen reference model
# ------------------------------------------------------------

reference = GPT.from_pretrained("gpt2")
reference.eval()
reference.to(device)

for param in reference.parameters():
    param.requires_grad = False


# ------------------------------------------------------------
# Reward head
# ------------------------------------------------------------

reward_head = nn.Linear(
    reference.config.n_embd,
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

for param in reward_head.parameters():
    param.requires_grad = False


# ------------------------------------------------------------
# Load prompts
# ------------------------------------------------------------

prompts = []

with prompts_path.open("r", encoding="utf-8") as f:
    for line in f:
        record = json.loads(line)
        prompts.append(record["prompt"])

print(f"Loaded {len(prompts)} prompts")


# ------------------------------------------------------------
# Sampling
# ------------------------------------------------------------

def generate_tokens(prompt):
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

    policy.eval()

    while x.size(1) < start_length + max_new_tokens:

        with torch.no_grad():
            logits, _ = policy(x)

        logits = logits[:, -1, :]
        logits = logits / temperature

        probs = F.softmax(logits, dim=-1)

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

        x = torch.cat((x, token), dim=1)

    return x, start_length


# ------------------------------------------------------------
# Reward
# ------------------------------------------------------------

def get_reward(x):
    with torch.no_grad():

        B, T = x.size()

        pos = torch.arange(
            0,
            T,
            dtype=torch.long,
            device=device,
        )

        tok_emb = reference.transformer.wte(x)
        pos_emb = reference.transformer.wpe(pos)

        h = tok_emb + pos_emb

        for block in reference.transformer.h:
            h = block(h)

        h = reference.transformer.ln_f(h)

        final_hidden = h[:, -1, :]

        reward = reward_head(final_hidden)

    return reward.squeeze()


# ------------------------------------------------------------
# Policy statistics
# ------------------------------------------------------------

def policy_stats(x, prompt_length):
    """
    Returns:

    mean log probability of generated tokens
    KL(policy || reference)
    """

    policy.train()

    policy_logits, _ = policy(x)

    with torch.no_grad():
        reference_logits, _ = reference(x)

    # Position p predicts token p+1.
    #
    # Generated tokens start at prompt_length,
    # so prediction starts at prompt_length - 1.

    policy_logits = policy_logits[
        :,
        prompt_length - 1:-1,
        :
    ]

    reference_logits = reference_logits[
        :,
        prompt_length - 1:-1,
        :
    ]

    targets = x[:, prompt_length:]

    policy_log_probs = F.log_softmax(
        policy_logits,
        dim=-1,
    )

    reference_log_probs = F.log_softmax(
        reference_logits,
        dim=-1,
    )

    # Log probability assigned to tokens we sampled
    token_log_probs = torch.gather(
        policy_log_probs,
        2,
        targets.unsqueeze(-1),
    ).squeeze(-1)

    sequence_log_prob = token_log_probs.mean()

    # True distribution KL at generated positions
    policy_probs = policy_log_probs.exp()

    kl = (
        policy_probs
        * (policy_log_probs - reference_log_probs)
    ).sum(dim=-1).mean()

    return sequence_log_prob, kl


# ------------------------------------------------------------
# Optimizer
# ------------------------------------------------------------

trainable_params = [
    p
    for p in policy.parameters()
    if p.requires_grad
]

print(
    "Trainable parameters:",
    f"{sum(p.numel() for p in trainable_params):,}",
)

optimizer = torch.optim.AdamW(
    trainable_params,
    lr=learning_rate,
)


# ------------------------------------------------------------
# Training
# ------------------------------------------------------------

for step in range(steps):

    prompt = random.choice(prompts)

    samples = []
    rewards = []

    # Generate multiple answers to the SAME prompt
    for _ in range(group_size):

        x, prompt_length = generate_tokens(prompt)

        reward = get_reward(x)

        samples.append(
            (x, prompt_length)
        )

        rewards.append(reward)

    rewards = torch.stack(rewards)

    # Relative reward within the group
    advantages = (
        rewards - rewards.mean()
    )

    # Optional normalization
    advantages = advantages / (
        rewards.std() + 1e-8
    )

    sequence_log_probs = []
    kls = []

    for x, prompt_length in samples:

        seq_log_prob, kl = policy_stats(
            x,
            prompt_length,
        )

        sequence_log_probs.append(seq_log_prob)
        kls.append(kl)

    sequence_log_probs = torch.stack(
        sequence_log_probs
    )

    kls = torch.stack(kls)

    # --------------------------------------------------------
    # CORRECT reward direction
    # --------------------------------------------------------

    policy_loss = -(
        advantages.detach()
        * sequence_log_probs
    ).mean()

    kl_loss = kl_beta * kls.mean()

    loss = policy_loss + kl_loss

    optimizer.zero_grad()
    loss.backward()

    torch.nn.utils.clip_grad_norm_(
        trainable_params,
        1.0,
    )

    optimizer.step()

    print(
        f"step {step + 1:03d} | "
        f"reward {rewards.mean().item():+.3f} | "
        f"min {rewards.min().item():+.3f} | "
        f"max {rewards.max().item():+.3f} | "
        f"kl {kls.mean().item():.4f} | "
        f"loss {loss.item():+.4f}"
    )


# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

output_path.parent.mkdir(
    parents=True,
    exist_ok=True,
)

torch.save(
    policy.state_dict(),
    output_path,
)

print(f"\nSaved policy to {output_path}")