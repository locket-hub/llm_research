import json
import random
from pathlib import Path

import tiktoken
from datasets import load_dataset


# ------------------------------------------------------------
# Settings
# ------------------------------------------------------------

num_prompts = 100
prompt_length = 50

output_path = Path("data/processed/prompts.jsonl")

rng = random.Random(42)


# ------------------------------------------------------------
# Paragraph filtering
# ------------------------------------------------------------

def valid_paragraph(text):
    text_lower = text.lower()

    bad_markers = [
        "http://",
        "https://",
        "www.",
        "isbn",
        "copyright",
        "all rights reserved",
        "publisher",
        "published by",
        "free-ebook",
        "click here",
        "<|endoftext|>",
    ]

    if any(marker in text_lower for marker in bad_markers):
        return False

    # Too short to provide useful prose
    if len(text) < 400:
        return False

    return True


# ------------------------------------------------------------
# Load dataset
# ------------------------------------------------------------

dataset = load_dataset(
    "lucadiliello/bookcorpusopen",
    split="train",
    streaming=True,
)

enc = tiktoken.get_encoding("gpt2")


# ------------------------------------------------------------
# Create output directory
# ------------------------------------------------------------

output_path.parent.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# Extract prompts
# ------------------------------------------------------------

prompts_written = 0

with output_path.open("w", encoding="utf-8") as f:

    for example in dataset:

        text = example["text"]

        paragraphs = text.split("\n\n")

        paragraphs = [
            " ".join(paragraph.split())
            for paragraph in paragraphs
        ]

        paragraphs = [
            paragraph
            for paragraph in paragraphs
            if valid_paragraph(paragraph)
        ]

        if not paragraphs:
            continue

        paragraph = rng.choice(paragraphs)

        tokens = enc.encode(
            paragraph,
            disallowed_special=(),
        )

        if len(tokens) < prompt_length:
            continue

        start = rng.randint(
            0,
            len(tokens) - prompt_length
        )

        prompt_tokens = tokens[start:start + prompt_length]
        prompt = enc.decode(prompt_tokens)

        record = {
            "id": prompts_written,
            "title": example["title"],
            "prompt": prompt,
        }

        f.write(json.dumps(record) + "\n")

        prompts_written += 1

        if prompts_written >= num_prompts:
            break


print(f"Saved {prompts_written} prompts to {output_path}")