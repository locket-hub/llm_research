Learned:

What you built

This is an attempt to recreate a known accident from OpenAI’s GPT-2 human-feedback work (RLHF, reinforcement learning from human feedback): a sign-flip bug in the reward led the model to be trained toward the text people liked least. This is the whole RLHF loop, on a small scale, on top of pretrained GPT-2 (124M):

Prompts: 100 short passages from BookCorpus.
Candidates: GPT-2 writes 4 completions per prompt.
Human labels: ranked the completions for 20 prompts.
Reward model: GPT-2 stays frozen, and a single linear layer is trained to score text the way you ranked it.
Policy training: GPT-2 is nudged toward completions that score highly, while a KL penalty holds it close to the original.
The bug: an identical copy of training with one minus sign added at line 334.
Evaluation: stock, normal and inverted models scored on the same prompts with the same seeds.
Concepts covered
Rankings become pairs. One ranking of 4 completions gives 6 “A beats B” comparisons.
Pairwise preference loss (Bradley–Terry): -log σ(r_better − r_worse). It only cares about which completion scores higher, not about absolute scores.
A reward model is a small layer on a language model. It reads GPT-2’s internal summary at the last token and outputs one number.
Policy gradient: raise the probability of samples that did better than expected, lower it for those that did worse.
Group-relative advantage: judge each completion against its siblings for the same prompt, instead of using an absolute reward.
KL penalty: keeps the policy writing like GPT-2 and stops it from gaming the reward model.
Steps vs. epochs: a step is one weight update; an epoch is one full pass over a fixed dataset. RL training counts steps because it generates new samples each time.
Inverted reward: only the reward is inverted, not the KL penalty. The model optimizes toward the worst text while still sounding fluent.
What the experiments showed
Run	Normal policy reward	Inverted policy reward	Stock GPT-2	KL from GPT-2
100 steps	−2.94	−1.96	−2.54	0.0003
500 steps	−2.40	−2.71	−2.54	0.0035
At 100 steps there was no real effect, and the result even pointed the wrong way.
At 500 steps the order was right (normal > stock > inverted), but the gap was within the noise.
The reward model reached 99% accuracy on its own training pairs but generalized only weakly. This is memorization: 769 trainable numbers against 120 comparisons.
Lessons
The reward model sets the upper limit. RL can only amplify the signal it’s given. With 20 labeled prompts there’s too little consistent signal, so a sign flip has little to flip.
High training accuracy can mean memorization. The useful check is how the model does on data it wasn’t trained on.
Normalizing by the group’s spread hides noise. Dividing by the standard deviation of 4 scores turns tiny random differences into full-strength updates.
Small settings add up to almost no movement. A learning rate of 1e-5, a loss averaged over 40 tokens, gradient clipping and only 2 of 12 blocks trainable led to KL ≈ 0.003 after 500 steps.
KL is the best gauge of how far the policy moved. The per-step reward printout is too noisy, because every step uses a different random prompt.
Small evaluations mislead. One sample per prompt over 50 prompts can’t detect a small effect, and different weights send the same seed down different text.
Stale checkpoints are easy to miss. Changing the data meant every downstream model had to be retrained.