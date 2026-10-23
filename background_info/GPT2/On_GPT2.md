# GPT-2: Notes and "Maximally-Bad Output"

## Part 1: How GPT-2 Works

GPT-2 is a **next-token predictor** built on a slightly modified transformer (from *Attention Is All You Need*).

### Pipeline

1. **Tokenization:** text (characters, words, numbers, phrases) is split into tokens, and each token gets an integer ID.
2. **Embeddings:** each token ID is turned into a vector by a learned token embedding. A learned position embedding is added so the model knows word order.
3. **Transformer blocks** (h1 → h2 → … → h12 in the 124M model). Each block has two parts:
   - **Attention:** where tokens *communicate*.
     - Multi-headed: several heads run in parallel, and their outputs are concatenated.
     - Each token makes a **Query**, **Key** and **Value**. Attention weights = softmax(Q·Kᵀ / √d), and those weights are applied to V.
     - The number of heads is folded into a batch dimension so all heads compute at once.
     - A **causal mask** (stored as `bias` in the code) stops tokens from attending to future tokens.
   - **MLP:** where each token *thinks individually* about what it gathered. It's two linear layers with a GELU between them. GPT-2 used the tanh approximation of GELU, and the exact version is standard now.
4. **Layer norm:** GPT-2 moved layer norm to the *start* of each sub-block (pre-norm) and added one final layer norm after the last block.
5. **LM head:** a linear layer (no bias) that projects from embedding size to vocab size, producing **logits**. Softmax turns logits into probabilities.

### In PyTorch

The model is an `nn.Module` containing:
- `wte`: token embedding (`nn.Embedding`)
- `wpe`: position embedding (`nn.Embedding`)
- `h`: a list of blocks, indexable by integer
- `ln_f`: the final layer norm
- `lm_head`: the projection to vocab size

A **tensor** is a multi-dimensional array.

### Feed-forward vs. recurrent networks

- **FFNN (feed-forward):** information flows input → output and never loops back.
- **RNN (recurrent):** outputs loop back in as inputs, so it processes sequences one step at a time. Transformers replaced RNNs for language because attention lets every position be processed in parallel.

### Training and generation

- **Training:** prefix text is encoded by the tokenizer into a tensor of token IDs, and the model learns to predict each next token.
- **Generation:**
  1. Run the model and keep only the **last position's logits** (the other positions aren't needed to predict the next token).
  2. Convert them to probabilities.
  3. Apply **top-k** (default k = 50): only the 50 most likely tokens can be picked.
  4. Sample one token, append it, and repeat.

---

## Part 2: What I Built

This is an attempt to recreate a known accident from OpenAI's GPT-2 human-feedback work (RLHF, reinforcement learning from human feedback). A sign-flip bug in the reward led the model to be trained toward the text people liked least.

It runs the whole RLHF loop, on a small scale, on top of pretrained GPT-2 (124M):

- **Prompts:** 100 short passages from BookCorpus.
- **Candidates:** GPT-2 writes 4 completions per prompt.
- **Human labels:** I ranked the completions for 20 prompts.
- **Reward model:** GPT-2 stays frozen, and a single linear layer is trained to score text the way I ranked it.
- **Policy training:** GPT-2 is nudged toward completions that score highly. A KL penalty holds it close to the original.
- **The bug:** an identical copy of training with one minus sign added at line 334.
- **Evaluation:** the stock, normal and inverted models are scored on the same prompts with the same seeds.

### Concepts Covered

- **Rankings become pairs.** One ranking of 4 completions gives 6 "A beats B" comparisons.
- **Pairwise preference loss (Bradley–Terry):** `-log σ(r_better − r_worse)`. It only cares about which completion scores higher, not about absolute scores.
- **A reward model is a small layer on a language model.** It reads GPT-2's internal summary at the last token and outputs one number.
- **Policy gradient:** raise the probability of samples that did better than expected, and lower it for those that did worse.
- **Group-relative advantage:** judge each completion against its siblings for the same prompt, instead of using an absolute reward.
- **KL penalty:** keeps the policy writing like GPT-2 and stops it from gaming the reward model.
- **Steps vs. epochs:** a step is one weight update, and an epoch is one full pass over a fixed dataset. RL training counts steps because it generates new samples each time.
- **Inverted reward:** only the reward is inverted, not the KL penalty. The model optimizes toward the worst text while still sounding fluent.

### What the Experiments Showed

| Run | Normal policy reward | Inverted policy reward | Stock GPT-2 | KL from GPT-2 |
|---|---|---|---|---|
| 100 steps | −2.94 | −1.96 | −2.54 | 0.0003 |
| 500 steps | −2.40 | −2.71 | −2.54 | 0.0035 |

- **100 steps:** there was no real effect, and the result even pointed the wrong way.
- **500 steps:** the order was right (normal > stock > inverted), but the gap was within the noise.
- **Reward model:** it reached 99% accuracy on its own training pairs but generalized only weakly. This is memorization: 769 trainable numbers against 120 comparisons.

### Lessons

1. **The reward model sets the upper limit.** RL can only amplify the signal it's given. With 20 labeled prompts there's too little consistent signal, so a sign flip has little to flip.
2. **High training accuracy can mean memorization.** The useful check is how the model does on data it wasn't trained on.
3. **Normalizing by the group's spread hides noise.** Dividing by the standard deviation of 4 scores turns tiny random differences into full-strength updates.
4. **Small settings add up to almost no movement.** A learning rate of 1e-5, a loss averaged over 40 tokens, gradient clipping and only 2 of 12 blocks trainable led to KL ≈ 0.003 after 500 steps.
5. **KL is the best gauge of how far the policy moved.** The per-step reward printout is too noisy, because every step uses a different random prompt.
6. **Small evaluations mislead.** One sample per prompt over 50 prompts can't detect a small effect, and different weights send the same seed down different text.
7. **Stale checkpoints are easy to miss.** Changing the data meant every downstream model had to be retrained.