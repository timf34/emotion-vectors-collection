# Emotion vectors collection

Linear emotion directions ("emotion vectors") in the residual stream of open-weight LLMs, extracted with the
method of Anthropic's *Emotion concepts in LLMs*: one vector per emotion word, from the mean activation over
stories in which a character feels that emotion.

**Vectors live on Hugging Face: [timf34/emotion-vectors](https://huggingface.co/datasets/timf34/emotion-vectors)**
(one folder per model). This repo has the index and a loader.

The denoised vectors (`vectors_{emotions,syndromes}_dn.pt`), plus `meta.json` and the held-out AUC CSVs, are also
in this repo under `vectors/<folder>/`. Raw vectors, neutral PCA bases and gguf control vectors are only on
[Hugging Face](https://huggingface.co/datasets/timf34/emotion-vectors).

## Models

| Folder | Model | Blocks × hidden | Stories read | Best layer (mean held-out AUC) |
|---|---|---|---|---|
| [`gemma3_27b`](https://huggingface.co/datasets/timf34/emotion-vectors/tree/main/gemma3_27b) | [google/gemma-3-27b-it](https://huggingface.co/google/gemma-3-27b-it) | 62 × 5376 | own | 52 (0.914) |
| [`gemma3_27b_pt`](https://huggingface.co/datasets/timf34/emotion-vectors/tree/main/gemma3_27b_pt) | [google/gemma-3-27b-pt](https://huggingface.co/google/gemma-3-27b-pt) (base) | 62 × 5376 | Gemma 3 IT's | 59 (0.918) |
| [`gemma3_27b_dpo`](https://huggingface.co/datasets/timf34/emotion-vectors/tree/main/gemma3_27b_dpo) | gemma-3-27b-it + [annasoli/gemma3-27b-dpo-calm-full](https://huggingface.co/annasoli/gemma3-27b-dpo-calm-full) (LoRA, merged) | 62 × 5376 | Gemma 3 IT's | 52 (0.914) |
| [`gemma3_27b_sft`](https://huggingface.co/datasets/timf34/emotion-vectors/tree/main/gemma3_27b_sft) | [annasoli/gemma3-27b-sft-diverse-calm-merged](https://huggingface.co/annasoli/gemma3-27b-sft-diverse-calm-merged) | 62 × 5376 | Gemma 3 IT's | 58 (0.920) |
| [`gemma3_27b_bct`](https://huggingface.co/datasets/timf34/emotion-vectors/tree/main/gemma3_27b_bct) | gemma-3-27b-it + [neilshah/bct-frustration-gemma3-27b](https://huggingface.co/neilshah/bct-frustration-gemma3-27b) (`epoch_1` LoRA, merged) | 62 × 5376 | Gemma 3 IT's | 59 (0.915) |
| [`gemma3_27b_no2733`](https://huggingface.co/datasets/timf34/emotion-vectors/tree/main/gemma3_27b_no2733) | gemma-3-27b-it with residual coordinate 2733 (a massive activation) zeroed at every layer | 62 × 5376 | Gemma 3 IT's | 38 (0.915) |
| [`gemma4_31b`](https://huggingface.co/datasets/timf34/emotion-vectors/tree/main/gemma4_31b) | [google/gemma-4-31B-it](https://huggingface.co/google/gemma-4-31B-it) | 60 × 5376 | own | 24 (0.952) |
| [`gemma2_27b`](https://huggingface.co/datasets/timf34/emotion-vectors/tree/main/gemma2_27b) | [google/gemma-2-27b-it](https://huggingface.co/google/gemma-2-27b-it) | 46 × 4608 | Gemma 3 IT's | 36 (0.921) |
| [`qwen3_32b`](https://huggingface.co/datasets/timf34/emotion-vectors/tree/main/qwen3_32b) | [Qwen/Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B) | 64 × 5120 | Gemma 3 IT's | 52 (0.924) |
| [`llama33_70b`](https://huggingface.co/datasets/timf34/emotion-vectors/tree/main/llama33_70b) | [meta-llama/Llama-3.3-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct) | 80 × 8192 | Gemma 3 IT's | 18 (0.928) |
| [`mistral_24b`](https://huggingface.co/datasets/timf34/emotion-vectors/tree/main/mistral_24b) | [mistralai/Mistral-Small-24B-Instruct-2501](https://huggingface.co/mistralai/Mistral-Small-24B-Instruct-2501) | 40 × 5120 | Gemma 3 IT's | 18 (0.926) |
| [`olmo2_32b`](https://huggingface.co/datasets/timf34/emotion-vectors/tree/main/olmo2_32b) | [allenai/OLMo-2-0325-32B-Instruct](https://huggingface.co/allenai/OLMo-2-0325-32B-Instruct) | 64 × 5120 | Gemma 3 IT's | 36 (0.927) |
| [`glm45_air`](https://huggingface.co/datasets/timf34/emotion-vectors/tree/main/glm45_air) | [zai-org/GLM-4.5-Air](https://huggingface.co/zai-org/GLM-4.5-Air) | 46 × 4096 | Gemma 3 IT's | 18 (0.924) |
| [`gptoss_20b`](https://huggingface.co/datasets/timf34/emotion-vectors/tree/main/gptoss_20b) | [openai/gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b) | 24 × 2880 | Gemma 3 IT's | 14 (0.914) |
| [`gptoss_120b`](https://huggingface.co/datasets/timf34/emotion-vectors/tree/main/gptoss_120b) | [openai/gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b) | 36 × 2880 | Gemma 3 IT's | 14 (0.916) |
| [`qwen36_27b`](https://huggingface.co/datasets/timf34/emotion-vectors/tree/main/qwen36_27b) | [Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) | 64 × 5120 | Gemma 3 IT's | 40 (0.925) |
| [`qwen38_27b`](https://huggingface.co/datasets/timf34/emotion-vectors/tree/main/qwen38_27b) | [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) | 64 × 5120 | Gemma 3 IT's | 40 (0.925) |

"Best layer" is the denoised layer with the highest mean one-vs-rest AUC across the 42 emotions on held-out
stories (chance = 0.5). Per-label, per-layer numbers are in each folder's `heldout_auc_emotions.csv`.

## Labels

**42 emotions:** depressed, sad, gloomy, melancholy, miserable, dispirited, listless, weary, worn out, tired,
resigned, lonely, heartbroken, grief-stricken, worthless, self-critical, ashamed, guilty, humiliated, embarrassed,
regretful, sorry, desperate, panicked, hysterical, overwhelmed, distressed, tormented, trapped, stuck, frustrated,
exasperated, angry, anxious, afraid, nervous, stressed, calm, self-confident, content, happy, hopeful.

**6 syndromes** (multi-sentence clusters): clinical_depression, acute_grief, burnout_exhaustion,
physical_illness_fatigue, anxiety_panic, frustration_blocked_goal.

## Files per folder

| File | Contents |
|---|---|
| `vectors_emotions_dn.pt` | `{emotion: {layer: Tensor[hidden]}}`, denoised, at the analysis layers (every 2nd block from 6, plus the last few). **Use these.** |
| `vectors_emotions_raw.pt` | same, not denoised, at every layer |
| `vectors_syndromes_{dn,raw}.pt` | same for the 6 syndromes |
| `neutral_pca.pt` | per-layer neutral-text principal components used for denoising |
| `heldout_auc_{emotions,syndromes}.csv` | held-out one-vs-rest AUC per label × layer × {raw, dn} |
| `meta.json` | HF model id, block count, hidden size, analysis layers, extraction config |
| `gguf/` | repeng-style control vectors (`direction.<layer>`) for llama.cpp / EasySteer (Gemma 3 / 4 and Qwen 3.6 / 3.8 folders) |

`gemma3_27b_no2733` has only the denoised emotion vectors. `gemma2_27b` and the other cross-family folders have no `gguf/`.

**Layer convention:** `layer = b` is the residual stream *after* decoder block `b` (0-based), i.e.
`hidden_states[b + 1]` in HuggingFace.

## Loading

```python
from load_vectors import load_vectors
vecs = load_vectors("qwen3_32b")            # {emotion: {layer: Tensor}}
v = vecs["depressed"][40]
```

From a clone of this repo: `torch.load("vectors/qwen3_32b/vectors_emotions_dn.pt")`.

Or directly: `huggingface_hub.hf_hub_download("timf34/emotion-vectors", "qwen3_32b/vectors_emotions_dn.pt", repo_type="dataset")`.

## Method

1. **Stories.** For each emotion, 100 topics × 6 stories of ~10 sentences in which a character feels it (plus syndrome
   and neutral sets). Gemma 3 27B IT and Gemma 4 31B IT wrote their own; every other model reads Gemma 3 IT's
   stories (the same text through different weights).
2. **Activations.** Mean residual activation over tokens from the 50th onward, per story, plain text (no chat
   template), bf16. The last 10% of each label's stories are held out.
3. **Vector** = label mean − mean across all labels in the set (emotions and syndromes centred separately).
4. **Denoising.** At the analysis layers, project out the top principal components of 40,000 neutral-story tokens
   that explain 50% of their variance.

Code: the `dprobe` package (`src/dprobe/extract.py`) in the emotion-concepts project.

**Caveat for Gemma 2/3:** their residual streams carry a few massive-activation coordinates (Gemma 3: dimension
2733). Cosine similarities between these vectors and other directions can be dominated by those coordinates;
drop or z-score them before comparing directions. Probe readouts and steering are unaffected.
