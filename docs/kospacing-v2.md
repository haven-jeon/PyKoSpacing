# KoSpacing v2 setup guide

KoSpacing v2 is the second spacing model of PyKoSpacing: Qwen3.5-0.8B fine-tuned to label every character of the
input (`Spacing(backend='v2')`). KoSpacing v1, the original TensorFlow model, stays the default.
This guide covers installing v2, pointing it at a checkpoint, running it on GPU or CPU, and, for maintainers,
publishing a checkpoint to the Hugging Face Hub.

The model is published on the Hugging Face Hub as **`gogamza/kotag-0.8b`**, a general instruction-following
sequence tagger; PyKoSpacing uses it for spacing under the name KoSpacing v2. It is the default checkpoint of
`backend='v2'`. (The repo is not created yet; until it is, pass a local checkpoint directory.)

## 1. Install

```bash
pip install "pykospacing[v2]"         # torch, transformers>=5.17, safetensors, huggingface_hub
```

- v2 does not import TensorFlow. Installing only the v2 extras is enough if you never use v1.
- On an NVIDIA GPU, also install the fast linear-attention kernels. Without them transformers runs Qwen3.5's
  linear-attention layers with slower reference code and prints a warning:

  ```bash
  pip install flash-linear-attention
  ```

- Tested with torch 2.14, transformers 5.17, huggingface_hub 1.31, Python 3.14.

## 2. Quick start

```python
from pykospacing import Spacing

spacing = Spacing(backend='v2')                  # gogamza/kotag-0.8b
spacing('아버지가방에들어가신다')                    # '아버지가 방에 들어가신다'
spacing.fix('오늘은날씨가 정말좋 네요.')             # '오늘은 날씨가 정말 좋네요.'
spacing(['밥먹었어', '오늘날씨가좋다'])               # a list is spaced as one batch
spacing.space_probs('아버지가방에')                 # [('아', 0.0), ..., ('가', 0.79), ...]
```

The first call downloads `gogamza/kotag-0.8b` (about 1.7 GB) to the Hugging Face cache; later calls load it from
there.

## 3. Options

| `Spacing(...)` argument | Environment variable | Default | Meaning |
|---|---|---|---|
| `backend='v2'` | | `'v1'` | Selects KoSpacing v2 |
| `checkpoint` | `PYKOSPACING_V2_CHECKPOINT` | `gogamza/kotag-0.8b` | Hugging Face repo id, or a local checkpoint directory |
| `revision` | `PYKOSPACING_V2_REVISION` | latest on `main` | Repo branch, tag or commit to use |
| `device` | `PYKOSPACING_V2_DEVICE` | `'cuda'` if available, else `'cpu'` | Where the model runs |
| `threads` | | torch default | CPU threads (CPU only) |
| `token_budget` | | 32768 | Padded tokens per forward pass; lower it if the GPU runs out of memory |
| `rules` | | `[]` | Words never split, as in v1 |

Arguments take precedence over environment variables. A value that starts with `.`, `/` or `~` is always a local
path; a missing directory raises `FileNotFoundError` instead of trying the Hub.

Pin a revision in production so a new upload does not change your results:

```python
spacing = Spacing(backend='v2', checkpoint='gogamza/kotag-0.8b', revision='v1.0')
```

### Methods

| Method | Input | Output |
|---|---|---|
| `spacing(text)` | text with or without spaces | Spaced text. Existing spaces are removed first and the text is spaced from scratch. |
| `spacing.fix(text)` | text whose spacing is partly wrong | Corrected text. Spaces the model agrees with are kept. |
| `spacing.space_probs(text)` | text | `[(char, P(space after char))]` over the non-space characters |

Each method also takes a list of strings, runs it as one batch and returns a list.
Inputs longer than 400 characters are split after a sentence end (or else a space) and the results are joined.
`ignore=` is a v1 option; v2 reads Latin letters, digits and symbols directly and raises `ValueError` if it is given.

## 4. Offline and private use

Download once, then load from the directory:

```bash
hf download gogamza/kotag-0.8b --revision v1.0 --local-dir ./kotag-0.8b
```

```python
spacing = Spacing(backend='v2', checkpoint='./kotag-0.8b')
```

Or keep using the repo id with the cache only: set `HF_HUB_OFFLINE=1`. If the checkpoint is not in the cache,
loading fails with `LocalEntryNotFoundError`. `HF_HOME` moves the cache.

For a private repo, log in first (`hf auth login`) or set `HF_TOKEN`.

## 5. GPU and CPU

| | GPU | CPU |
|---|---|---|
| Precision | bfloat16 | float32 |
| Weights in memory | about 1.7 GB | about 3.4 GB |
| One sentence of about 45 characters | about 18 ms | about 1.1 s |
| A list in one call | about 70 sentences/s | about 1 sentence/s |

- On CPU, v2 is about 100 times slower than v1. Use a GPU for volume work, and pass lists instead of looping over
  sentences.
- On CPU, more threads barely help (8 threads: 1,150 ms, 20 threads: 1,120 ms on a 20-core Arm CPU).
- CPU and GPU agree on almost every sentence (49 of 50 in a test). The differences are borderline spaces, caused by
  float32 vs bfloat16.
- If `flash-linear-attention` is installed, transformers uses its GPU-only kernels even for a model on the CPU.
  PyKoSpacing hides that package when it builds a CPU model. For this to work, the CPU model must be created before
  any GPU model in the same process; otherwise you get a `RuntimeError`.

## 6. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `RepositoryNotFoundError` for `gogamza/kotag-0.8b` | The repo is not published yet, or is private: pass a local `checkpoint=` directory, or log in (section 4). |
| `LocalEntryNotFoundError` | `HF_HUB_OFFLINE=1` and the checkpoint is not cached: download it first (section 4). |
| `RuntimeError: 0 active drivers` (Triton) | A CPU model was built after the GPU kernels were bound: build the CPU model first, or use a separate process. |
| `ModuleNotFoundError: ...qwen3_5` | transformers is too old for Qwen3.5: `pip install "transformers>=5.17"`. |
| Warning `... falling back to its reference PyTorch implementation` on GPU | Fast kernels are missing: `pip install flash-linear-attention` (section 1). |
| GPU out of memory | Lower `token_budget`, e.g. `Spacing(backend='v2', checkpoint=..., token_budget=8192)`. |
| `NotImplementedError: fix() needs KoSpacing v2` | `fix()` and `space_probs()` exist only for `backend='v2'`. |

## 7. For maintainers: publishing a checkpoint

Checkpoints are trained in the kospacing project. The repo's own files (model card template, `kotag.py`,
`PROMPT.md`, `examples/`) live in kospacing `hf/kotag-0.8b/`, and `scripts/export_hf.py` combines them with a
checkpoint into the folder to upload.

### 7.1 Files

| File | Content |
|---|---|
| `config.json`, `model.safetensors` | Qwen3.5 backbone (`Qwen3_5Model`) |
| `readout.safetensors` | `Linear(hidden, 255)` readout over single-token option codes |
| `decision_config.json` | option codes and their token ids (`format_version` 1) |
| `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja` | tokenizer and chat template |
| `README.md` | model card, rendered from the template in kospacing `hf/kotag-0.8b/README.md.tmpl` |
| `kotag.py`, `PROMPT.md`, `examples/` | standalone inference code for any tagging question, prompt format, examples |

`pykospacing/v2.py` reads exactly these files. It refuses a `decision_config.json` whose `format_version` it does not
know, so a new format needs a new PyKoSpacing release.

### 7.2 Before uploading

1. **Choose the checkpoint and measure it.** In kospacing, run `scripts/eval_pykospacing.py`,
   `scripts/eval_pykospacing_fix.py` (and with `--data data/processed/eval_colloq.jsonl`) with `--backend v2`, plus
   `scripts/bench_pykospacing.py`. The model card takes its numbers from these result files.
2. **Decide the license.** The base model, Qwen3.5-0.8B, is Apache-2.0. The training data must allow redistribution
   of a model trained on it:
   - Korean Wikipedia, KLUE-NER, UD Korean: CC BY-SA 4.0. RAGTruth: MIT.
   - KOLD states no license. Ask its authors, or train without KOLD.
   - Some task rows were generated by google/gemma-4-26B-A4B-it. Check that Gemma's terms allow training and
     releasing a model on its outputs.
3. **Export and check** (in kospacing):

   ```bash
   uv run --project jeff python -I scripts/export_hf.py --checkpoint runs/<run>/final --tag <run> \
       --repo-id gogamza/kotag-0.8b --license <license> --output exports/kotag-0.8b
   ```

   The script replays the run's sampling to fill the task table, loads the exported folder with its own
   `kotag.py` and with PyKoSpacing, and stops if either spaces differently from the source checkpoint.
   Read the generated `README.md`; results not measured yet are shown as `–`.

### 7.3 Upload

```bash
hf auth login                                                     # a token with write access
hf repos create gogamza/kotag-0.8b --private                   # private until checked
hf upload gogamza/kotag-0.8b exports/kotag-0.8b . --commit-message "kotag-0.8b (<run>)"
hf repos tag create gogamza/kotag-0.8b v1.0 -m "<run>"         # the revision users pin
```

Then check it from a clean environment:

```bash
python -c "from pykospacing import Spacing; print(Spacing(backend='v2', checkpoint='gogamza/kotag-0.8b', revision='v1.0')('아버지가방에들어가신다'))"
```

Make the repo public with `hf repos settings gogamza/kotag-0.8b --no-private` once it works.

### 7.4 Versioning

- Each released checkpoint gets a new tag (`v1.0`, `v1.1`, ...). The tags version kotag-0.8b itself, not the
  KoSpacing v1/v2 naming. Never move a tag; users pin them.
- Upload a new checkpoint to `main` and tag it. Users who do not pin a revision get the newest one.
- If you change the file format, bump `format_version` and release a PyKoSpacing version that reads it, before
  you upload.
