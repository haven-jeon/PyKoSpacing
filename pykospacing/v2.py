# -*- coding: utf-8 -*-
"""KoSpacing v2: an instruction-following tag model (Qwen3.5-0.8B with a per-character readout, kospacing project).

One model labels every character of the input in a single forward pass. The spacing and spacing-correction prompts
below are fixed wordings the model was trained on, so results match the "seen" columns of the kospacing evaluation.

Needs torch/transformers and the kospacing code (kotag, which loads jeff); TensorFlow is not used. The code is found
through the KOSPACING_ROOT environment variable (default ~/work/kospacing) until the inference code is packaged.
"""

import os
import re
import sys
from pathlib import Path

__all__ = ['KoSpacingV2', 'SPACING_QUESTION', 'FIX_QUESTION']

SPACING_QUESTION = {"type": "tag", "unit": "char", "instructions": "띄어쓰기를 해라",
                    "criteria": {"0": "다음 글자와 붙여 씀", "1": "이 글자 뒤에서 띄어 씀"}}
FIX_QUESTION = {"type": "tag", "unit": "char", "instructions": "잘못된 띄어쓰기를 바로잡아라",
                "criteria": {"K": "이 글자 뒤의 공백 상태를 그대로 둠", "I": "이 글자 뒤에 공백을 넣음",
                             "D": "이 글자 뒤의 공백을 지움"}}
MAX_CHARS = 400          # characters per model input; longer lines are split at sentence ends, then spaces
SENTENCE_END = re.compile(r'[.!?…。]+["\'”’)\]]*')


def _import_tag_model():
    try:
        from kotag.model import TagModel
    except ImportError:
        root = Path(os.environ.get('KOSPACING_ROOT', Path.home() / 'work' / 'kospacing')).expanduser()
        if not (root / 'kotag' / 'model.py').exists():
            raise ImportError(f"KoSpacing v2: kotag not found; set KOSPACING_ROOT (tried {root})") from None
        sys.path.insert(0, str(root))
        from kotag.model import TagModel
    return TagModel


QWEN_MODELING = 'transformers.models.qwen3_5.modeling_qwen3_5'


def _block_gpu_kernels():
    """transformers binds Qwen3.5's linear-attention functions to flash-linear-attention (Triton, GPU only) when the
    `fla` package is importable, whatever the device. On CPU, hide `fla` before the modeling module is imported so the
    PyTorch implementations are used."""
    if QWEN_MODELING in sys.modules:
        if sys.modules.get('fla') is not None:
            raise RuntimeError("KoSpacing v2: create the CPU model before any GPU model in this process "
                               "(the GPU kernels are already bound)")
        return
    sys.modules['fla'] = None


def split_long(text, limit=MAX_CHARS):
    """Pieces of at most `limit` characters, cut after a sentence end when possible, else after a space."""
    pieces = []
    while len(text) > limit:
        window = text[:limit]
        cut = max((m.end() for m in SENTENCE_END.finditer(window)), default=0)
        if cut < limit // 4:
            cut = window.rfind(' ') + 1
        if cut < limit // 4:
            cut = limit
        pieces.append(text[:cut])
        text = text[cut:]
    if text:
        pieces.append(text)
    return pieces


class KoSpacingV2:
    def __init__(self, checkpoint=None, device=None, token_budget=32768):
        checkpoint = checkpoint or os.environ.get('PYKOSPACING_V2_CHECKPOINT')
        if not checkpoint:
            raise ValueError("KoSpacing v2: pass checkpoint= or set PYKOSPACING_V2_CHECKPOINT")
        import torch
        device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        if device == 'cpu':
            _block_gpu_kernels()
        TagModel = _import_tag_model()
        self.model = TagModel(str(checkpoint), device=device, train=False)
        self.token_budget = token_budget

    def _predict(self, texts, question):
        rows = [{"id": str(i), "state": t, "question": question} for i, t in enumerate(texts)]
        keys = list(question["criteria"])
        return [[dict(zip(keys, p)) for p in unit] for unit in self.model.predict(rows, self.token_budget)]

    def space_probs(self, texts):
        """Per text: list of (character, probability that a space follows) over its non-space characters."""
        pieces, owner = [], []
        for index, text in enumerate(texts):
            for piece in split_long("".join(text.split())):
                pieces.append(piece)
                owner.append(index)
        results = [[] for _ in texts]
        for index, piece, probs in zip(owner, pieces, self._predict(pieces, SPACING_QUESTION) if pieces else []):
            results[index] += [(c, p["1"]) for c, p in zip(piece, probs)]
        return results

    def space(self, texts, threshold=0.5):
        out = []
        for probs in self.space_probs(texts):
            out.append("".join(c + (' ' if p > threshold else '') for c, p in probs).strip())
        return out

    def fix(self, texts):
        """Correct the spacing of texts that already have spaces; spaces the model does not object to are kept."""
        pieces, owner, spaced_after = [], [], []
        for index, text in enumerate(texts):
            parts = split_long(" ".join(text.split()))
            for k, part in enumerate(parts):
                if part.strip():
                    pieces.append(part.strip())
                    owner.append(index)
                    # the space (or none) at a cut between pieces is kept as written
                    spaced_after.append(part.endswith(' ') or (k + 1 < len(parts) and parts[k + 1].startswith(' ')))
        chars = [[] for _ in texts]
        predictions = self._predict(pieces, FIX_QUESTION) if pieces else []
        for index, piece, after, probs in zip(owner, pieces, spaced_after, predictions):
            gaps = [i + 1 < len(piece) and piece[i + 1] == ' ' for i, c in enumerate(piece) if c != ' ']
            labels = [max(p, key=p.get) for p in probs]
            out = [c + (' ' if (label == 'I' or (gap and label != 'D')) else '')
                   for c, gap, label in zip(piece.replace(' ', ''), gaps, labels)]
            out[-1] = out[-1].rstrip() + (' ' if after else '')
            chars[index] += out
        return [re.sub(r' +', ' ', "".join(c)).strip() for c in chars]
