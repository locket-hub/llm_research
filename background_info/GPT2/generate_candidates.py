import json
from pathlib import Path

import torch
import tiktoken

from model import GPT


# ------------------------------------------------------------
# Settings
# ------------------------------------------------------------

input_path = Path("data/processed/prompts.jsonl")
output_path = Path("data/processed/candidates.jsonl")

num_candidates = 4
max_new_tokens = 35
temperature = 1.1
top_k = 100


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
# Load model + tokenizer
# ------------------------------------------------------------

model = GPT.from_pretrained("gpt2")
model.eval()
model.to(device)

enc = tiktoken.get_encoding("gpt2")


# ------------------------------------------------------------
# Generation function
# ------------------------------------------------------------

def generate(prompt):
    tokens = enc.encode(prompt)

    x = torch.tensor(tokens, dtype=torch.long)
    x = x.unsqueeze(0).to(device)

    start_length = x.size(1)

    while x.size(1) < start_length + max_new_tokens:

        with torch.no_grad():
            logits, _ = model(x)

        logits = logits[:, -1, :]
        logits = logits / temperature

        probs = torch.softmax(logits, dim=-1)

        topk_probs, topk_indices = torch.topk(
            probs,
            top_k,
            dim=-1,
        )

        ix = torch.multinomial(topk_probs, num_samples=1)
        xcol = torch.gather(topk_indices, -1, ix)

        x = torch.cat((x, xcol), dim=1)

    generated_tokens = x[0, start_length:].tolist()

    return enc.decode(generated_tokens)


# ------------------------------------------------------------
# Generate candidates
# ------------------------------------------------------------

output_path.parent.mkdir(parents=True, exist_ok=True)

with input_path.open("r", encoding="utf-8") as infile, \
     output_path.open("w", encoding="utf-8") as outfile:

    for line in infile:

        record = json.loads(line)

        prompt = record["prompt"]

        candidates = []

        for i in range(num_candidates):
            completion = generate(prompt)
            candidates.append(completion)

        output_record = {
            "id": record["id"],
            "title": record["title"],
            "prompt": prompt,
            "candidates": candidates,
        }

        outfile.write(json.dumps(output_record) + "\n")

        print(f"Generated candidates for prompt {record['id']}")