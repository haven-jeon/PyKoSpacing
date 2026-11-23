"""Korean jamo / confusable / josa spell-correction candidate generator (CPU, zero GPU).

Provides:
- Hangul syllable decomposition & composition (`decompose`, `compose`, `has_final_consonant`)
- 2-beolsik keyboard neighbor graph (`KEYBOARD_NEIGHBORS`)
- High-recall Korean confusable rules & particle-batchim harmony fixes (`CONFUSABLE_RULES`, `fix_josa_candidates`)
- SymSpell-style jamo delete-1 index over train-only word & stem frequency vocabularies (`SpellCandidateGenerator`)
"""

import gzip
import json
import math
import re
from collections import defaultdict
from pathlib import Path

CHOSUNG = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
JUNGSUNG = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
JONGSUNG = [
    "", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ", "ㄻ", "ㄼ", "ㄽ", "ㄾ", "ㄿ", "ㅀ",
    "ㅁ", "ㅂ", "ㅄ", "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ",
]

CHO_IDX = {c: i for i, c in enumerate(CHOSUNG)}
JUNG_IDX = {c: i for i, c in enumerate(JUNGSUNG)}
JONG_IDX = {c: i for i, c in enumerate(JONGSUNG) if c}

# Split compound jongsung / jungsung into atomic jamo for closer keyboard edit distances when decomposing to flat jamo
COMPOUND_JONG_SPLIT = {
    "ㄳ": "ㄱㅅ", "ㄵ": "ㄴㅈ", "ㄶ": "ㄴㅎ", "ㄺ": "ㄹㄱ", "ㄻ": "ㄹㅁ", "ㄼ": "ㄹㅂ",
    "ㄽ": "ㄹㅅ", "ㄾ": "ㄹㅌ", "ㄿ": "ㄹㅍ", "ㅀ": "ㄹㅎ", "ㅄ": "ㅂㅅ",
}
COMPOUND_JUNG_SPLIT = {
    "ㅘ": "ㅗㅏ", "ㅙ": "ㅗㅐ", "ㅚ": "ㅗㅣ", "ㅝ": "ㅜㅓ", "ㅞ": "ㅜㅔ", "ㅟ": "ㅜㅣ", "ㅢ": "ㅡㅣ",
}

HANGUL_SYLLABLE = re.compile(r"[가-힣]")
PURE_HANGUL_WORD = re.compile(r"[가-힣]+")
PURE_ASCII_ALPHA = re.compile(r"[A-Za-z]+")
ASCII_ALPHA_RE = re.compile(r"[A-Za-z]")
HANGUL_JAMO_RE = re.compile(r"[ㄱ-ㅎㅏ-ㅣ]")

# Unified 2-beolsik Korean + QWERTY English / digit physical keyboard layout (row, staggered col)
_HANGUL_KEY_POS: dict[str, tuple[float, float]] = {
    "ㅂ": (0, 0.0), "ㅃ": (0, 0.0), "ㅈ": (0, 1.0), "ㅉ": (0, 1.0), "ㄷ": (0, 2.0), "ㄸ": (0, 2.0),
    "ㄱ": (0, 3.0), "ㄲ": (0, 3.0), "ㅅ": (0, 4.0), "ㅆ": (0, 4.0),
    "ㅛ": (0, 5.0), "ㅕ": (0, 6.0), "ㅑ": (0, 7.0), "ㅐ": (0, 8.0), "ㅒ": (0, 8.0), "ㅔ": (0, 9.0), "ㅖ": (0, 9.0),
    "ㅁ": (1, 0.25), "ㄴ": (1, 1.25), "ㅇ": (1, 2.25), "ㄹ": (1, 3.25), "ㅎ": (1, 4.25),
    "ㅗ": (1, 5.25), "ㅓ": (1, 6.25), "ㅏ": (1, 7.25), "ㅣ": (1, 8.25),
    "ㅋ": (2, 0.75), "ㅌ": (2, 1.75), "ㅊ": (2, 2.75), "ㅍ": (2, 3.75),
    "ㅠ": (2, 4.75), "ㅜ": (2, 5.75), "ㅡ": (2, 6.75),
}

_BASE_QWERTY_TO_JAMO: dict[str, str] = {
    "q": "ㅂ", "w": "ㅈ", "e": "ㄷ", "r": "ㄱ", "t": "ㅅ", "y": "ㅛ", "u": "ㅕ", "i": "ㅑ", "o": "ㅐ", "p": "ㅔ",
    "a": "ㅁ", "s": "ㄴ", "d": "ㅇ", "f": "ㄹ", "g": "ㅎ", "h": "ㅗ", "j": "ㅓ", "k": "ㅏ", "l": "ㅣ",
    "z": "ㅋ", "x": "ㅌ", "c": "ㅊ", "v": "ㅍ", "b": "ㅠ", "n": "ㅜ", "m": "ㅡ",
}
_SHIFT_QWERTY_TO_JAMO: dict[str, str] = {
    "Q": "ㅃ", "W": "ㅉ", "E": "ㄸ", "R": "ㄲ", "T": "ㅆ", "O": "ㅒ", "P": "ㅖ",
}
QWERTY_TO_JAMO: dict[str, str] = {
    **_BASE_QWERTY_TO_JAMO,
    **{k.upper(): _SHIFT_QWERTY_TO_JAMO.get(k.upper(), v) for k, v in _BASE_QWERTY_TO_JAMO.items()},
}
JAMO_TO_QWERTY: dict[str, str] = {
    **{v: k for k, v in _BASE_QWERTY_TO_JAMO.items()},
    **{v: k for k, v in _SHIFT_QWERTY_TO_JAMO.items()},
}

_ASCII_KEY_POS: dict[str, tuple[float, float]] = {
    **{ch: (-1.0, idx - 0.5) for idx, ch in enumerate("1234567890")},
    **{ch: _HANGUL_KEY_POS[jamo] for ch, jamo in QWERTY_TO_JAMO.items()},
}
_KEY_POS: dict[str, tuple[float, float]] = {**_HANGUL_KEY_POS, **_ASCII_KEY_POS}

CONSONANTS = set(CHOSUNG)
VOWELS = set(JUNGSUNG)

# Phonetic / tense / same-key shift near-miss pairs (kept for backwards compatibility)
SOFT_CONFUSABLE_PAIRS: set[tuple[str, str]] = {
    ("ㅐ", "ㅔ"), ("ㅔ", "ㅐ"), ("ㅒ", "ㅖ"), ("ㅖ", "ㅒ"),
    ("ㅐ", "ㅒ"), ("ㅒ", "ㅐ"), ("ㅔ", "ㅖ"), ("ㅖ", "ㅔ"),
    ("ㄱ", "ㄲ"), ("ㄲ", "ㄱ"), ("ㄷ", "ㄸ"), ("ㄸ", "ㄷ"),
    ("ㅂ", "ㅃ"), ("ㅃ", "ㅂ"), ("ㅅ", "ㅆ"), ("ㅆ", "ㅅ"), ("ㅈ", "ㅉ"), ("ㅉ", "ㅈ"),
}

# Asymmetric empirical phonetic / orthographic substitution costs calibrated from
# Jeon, Huang & Rim (2010) Table 2 (2,252 labeled query typo pairs) & Table 4 (100M logs):
# maps (typed_jamo, intended_jamo) -> calibrated cost in [0.42, 0.64].
_PHONETIC_SUB_COST: dict[tuple[str, str], float] = {
    # Table 2 high-probability directional pairs (P >= 0.75)
    ("ㅆ", "ㅅ"): 0.42, ("ㅅ", "ㅆ"): 0.52,
    ("ㅠ", "ㅜ"): 0.44, ("ㅜ", "ㅠ"): 0.58,
    ("ㅐ", "ㅔ"): 0.44, ("ㅔ", "ㅐ"): 0.48,
    ("ㄸ", "ㅌ"): 0.46, ("ㅌ", "ㄸ"): 0.62,
    ("ㅕ", "ㅓ"): 0.46, ("ㅓ", "ㅕ"): 0.58,
    ("ㅑ", "ㅏ"): 0.46, ("ㅏ", "ㅑ"): 0.58,
    ("ㄲ", "ㅋ"): 0.46, ("ㅋ", "ㄲ"): 0.62,
    ("ㅖ", "ㅔ"): 0.46, ("ㅔ", "ㅖ"): 0.55,
    ("ㅎ", "ㅇ"): 0.48, ("ㅇ", "ㅎ"): 0.62,
    # Table 2 & Table 4 medium/high phonetic, loanword, liquid/nasal & tense pairs
    ("ㅛ", "ㅗ"): 0.48, ("ㅗ", "ㅛ"): 0.58,
    ("ㅍ", "ㅂ"): 0.50, ("ㅂ", "ㅍ"): 0.58,
    ("ㅃ", "ㅍ"): 0.52, ("ㅍ", "ㅃ"): 0.64,
    ("ㅊ", "ㅈ"): 0.52, ("ㅈ", "ㅊ"): 0.58,
    ("ㅋ", "ㄱ"): 0.54, ("ㄱ", "ㅋ"): 0.58,
    ("ㅌ", "ㄷ"): 0.54, ("ㄷ", "ㅌ"): 0.58,
    ("ㅒ", "ㅐ"): 0.52, ("ㅐ", "ㅒ"): 0.56,
    ("ㅒ", "ㅖ"): 0.52, ("ㅖ", "ㅒ"): 0.54,
    ("ㅡ", "ㅜ"): 0.52, ("ㅜ", "ㅡ"): 0.54,
    ("ㄲ", "ㄱ"): 0.48, ("ㄱ", "ㄲ"): 0.52,
    ("ㄸ", "ㄷ"): 0.48, ("ㄷ", "ㄸ"): 0.52,
    ("ㅃ", "ㅂ"): 0.48, ("ㅂ", "ㅃ"): 0.52,
    ("ㅉ", "ㅈ"): 0.48, ("ㅈ", "ㅉ"): 0.52,
    ("ㅉ", "ㅊ"): 0.54, ("ㅊ", "ㅉ"): 0.64,
    ("ㅓ", "ㅏ"): 0.56,
    ("ㅓ", "ㅗ"): 0.58, ("ㅗ", "ㅓ"): 0.58,
    ("ㄴ", "ㄹ"): 0.62, ("ㄹ", "ㄴ"): 0.62,
}


def _physical_keyboard_cost(a: str, b: str) -> float:
    """Physical 2-beolsik / QWERTY coordinate distance cost between two atomic keys."""
    if a == b:
        return 0.0
    if a.isascii() and b.isascii() and a.lower() == b.lower():
        return 0.15
    p1 = _KEY_POS.get(a)
    p2 = _KEY_POS.get(b)
    if p1 is None or p2 is None:
        return 1.00
    dist = math.hypot(p1[0] - p2[0], p1[1] - p2[1])
    if dist == 0.0:
        return 0.50
    if dist <= 1.15:
        return 0.60
    if dist <= 1.40:
        return 0.75
    if dist <= 2.05:
        return 0.92
    return 1.00


def keyboard_sub_cost(a: str, b: str) -> float:
    """Asymmetric physical keyboard + empirical phonetic substitution cost (`a`=typed -> `b`=intended)."""
    if a == b:
        return 0.0
    phys_cost = _physical_keyboard_cost(a, b)
    phon_cost = _PHONETIC_SUB_COST.get((a, b), 1.00)
    return min(phys_cost, phon_cost)


_SUB_COST: dict[tuple[str, str], float] = {
    (k1, k2): keyboard_sub_cost(k1, k2)
    for k1 in set(_KEY_POS) | {a for a, _ in _PHONETIC_SUB_COST} | {b for _, b in _PHONETIC_SUB_COST}
    for k2 in set(_KEY_POS) | {a for a, _ in _PHONETIC_SUB_COST} | {b for _, b in _PHONETIC_SUB_COST}
    if k1 != k2
}


def _build_keyboard_neighbors() -> dict[str, tuple[str, ...]]:
    neighbors: dict[str, list[str]] = defaultdict(list)
    keys = list(_HANGUL_KEY_POS)
    for k1 in keys:
        r1, c1 = _HANGUL_KEY_POS[k1]
        same_kind = CONSONANTS if k1 in CONSONANTS else VOWELS
        for k2 in keys:
            if k1 == k2 or k2 not in same_kind:
                continue
            r2, c2 = _HANGUL_KEY_POS[k2]
            if math.hypot(r1 - r2, c1 - c2) <= 1.35:
                neighbors[k1].append(k2)
    # Center consonant<->vowel keyboard boundary pairs + common vowel confusions + ASCII QWERTY neighbors
    extra_pairs = [
        ("ㅐ", "ㅔ"), ("ㅒ", "ㅖ"), ("ㅚ", "ㅙ"), ("ㅙ", "ㅞ"), ("ㅚ", "ㅞ"),
        ("ㅅ", "ㅛ"), ("ㅎ", "ㅗ"), ("ㅍ", "ㅠ"),
    ]
    for a, b in extra_pairs:
        if b not in neighbors[a]:
            neighbors[a].append(b)
        if a not in neighbors[b]:
            neighbors[b].append(a)
    ascii_lower = [k for k in _ASCII_KEY_POS if k.islower() or k.isdigit()]
    for k1 in ascii_lower:
        r1, c1 = _ASCII_KEY_POS[k1]
        for k2 in ascii_lower:
            if k1 != k2 and math.hypot(r1 - r2, c1 - c2) <= 1.35:
                neighbors[k1].append(k2)
    return {k: tuple(dict.fromkeys(v)) for k, v in neighbors.items()}


def _build_phonetic_neighbors() -> dict[str, tuple[str, ...]]:
    phon: dict[str, list[str]] = defaultdict(list)
    for (src, dst), cost in sorted(_PHONETIC_SUB_COST.items(), key=lambda kv: (kv[1], kv[0])):
        if cost <= 0.62:
            phon[src].append(dst)
    return {k: tuple(dict.fromkeys(v)) for k, v in phon.items()}


KEYBOARD_NEIGHBORS: dict[str, tuple[str, ...]] = _build_keyboard_neighbors()
KEYBOARD_NEIGHBOR_SET: dict[str, set[str]] = {k: set(v) for k, v in KEYBOARD_NEIGHBORS.items()}
PHONETIC_NEIGHBORS: dict[str, tuple[str, ...]] = _build_phonetic_neighbors()
_LOOKUP_SUB_ALTS: dict[str, tuple[str, ...]] = {
    ch: tuple(dict.fromkeys([*KEYBOARD_NEIGHBORS.get(ch, ()), *PHONETIC_NEIGHBORS.get(ch, ())]))
    for ch in set(KEYBOARD_NEIGHBORS) | set(PHONETIC_NEIGHBORS)
}


def is_hangul_syllable(ch: str) -> bool:
    return len(ch) == 1 and "가" <= ch <= "힣"


def decompose_syllable(ch: str) -> tuple[int, int, int] | None:
    if not is_hangul_syllable(ch):
        return None
    code = ord(ch) - 0xAC00
    return code // 588, (code % 588) // 28, code % 28


def compose_syllable(cho: int, jung: int, jong: int = 0) -> str:
    return chr(0xAC00 + cho * 588 + jung * 28 + jong)


def has_final_consonant(ch: str) -> bool:
    parts = decompose_syllable(ch)
    return parts is not None and parts[2] != 0


def jongsung_of(ch: str) -> str:
    parts = decompose_syllable(ch)
    return JONGSUNG[parts[2]] if parts is not None else ""


def decompose(text: str, split_compound: bool = True) -> str:
    """Decompose Hangul syllables in `text` into atomic jamo characters."""
    out: list[str] = []
    for ch in text:
        parts = decompose_syllable(ch)
        if parts is None:
            out.append(ch)
            continue
        c, v, f = parts
        out.append(CHOSUNG[c])
        jung = JUNGSUNG[v]
        out.append(COMPOUND_JUNG_SPLIT.get(jung, jung) if split_compound else jung)
        if f:
            jong = JONGSUNG[f]
            out.append(COMPOUND_JONG_SPLIT.get(jong, jong) if split_compound else jong)
    return "".join(out)


_REV_JUNG_SPLIT = {v: k for k, v in COMPOUND_JUNG_SPLIT.items()}
_REV_JONG_SPLIT = {v: k for k, v in COMPOUND_JONG_SPLIT.items()}


def compose(jamo_str: str) -> str:
    """Compose atomic jamo sequence back into Hangul syllables where valid."""
    out: list[str] = []
    i = 0
    n = len(jamo_str)
    while i < n:
        ch = jamo_str[i]
        if ch in CHO_IDX and i + 1 < n and jamo_str[i + 1] in JUNG_IDX:
            cho = CHO_IDX[ch]
            i += 1
            # Check compound vowel
            if i + 1 < n and jamo_str[i:i + 2] in _REV_JUNG_SPLIT:
                jung_ch = _REV_JUNG_SPLIT[jamo_str[i:i + 2]]
                i += 2
            else:
                jung_ch = jamo_str[i]
                i += 1
            jung = JUNG_IDX[jung_ch]
            jong = 0
            # Check trailing consonant(s) not followed by a vowel
            if i < n and jamo_str[i] in JONG_IDX:
                # Look ahead for compound jongsung first
                if (
                    i + 1 < n
                    and jamo_str[i:i + 2] in _REV_JONG_SPLIT
                    and (i + 2 >= n or jamo_str[i + 2] not in JUNG_IDX)
                ):
                    jong = JONG_IDX[_REV_JONG_SPLIT[jamo_str[i:i + 2]]]
                    i += 2
                elif i + 1 >= n or jamo_str[i + 1] not in JUNG_IDX:
                    jong = JONG_IDX[jamo_str[i]]
                    i += 1
            out.append(compose_syllable(cho, jung, jong))
        else:
            out.append(ch)
            i += 1
    return "".join(out)


# Bidirectional & directional frequent Korean confusable rules (applied on whole word or substrings)
CONFUSABLE_SUBSTRING_RULES: list[tuple[str, str]] = [
    # 되 / 돼
    ("되요", "돼요"), ("돼요", "되요"),
    ("됬", "됐"), ("됐", "됬"),
    ("되서", "돼서"), ("돼서", "되서"),
    ("되야", "돼야"), ("돼야", "되야"),
    ("안되", "안돼"), ("안돼", "안되"),
    ("돼는", "되는"), ("돼지", "되지"), ("돼고", "되고"), ("돼면", "되면"), ("돼었", "되었"),
    ("되는", "돼는"), ("되지", "돼지"), ("되고", "돼고"), ("되면", "돼면"), ("되었", "돼었"),
    ("돼나", "되나"), ("되나", "돼나"),
    # 안 / 않
    ("않돼", "안돼"), ("않되", "안되"), ("않해", "안해"), ("않먹", "안먹"), ("않받", "안받"),
    ("않보", "안보"), ("않가", "안가"), ("않오", "안오"), ("않하", "안하"),
    ("안아", "않아"), ("안고", "않고"), ("안는", "않는"), ("안았", "않았"), ("안은", "않은"),
    ("않아", "안아"), ("않고", "안고"), ("않는", "안는"), ("않았", "안았"),
    # 던 / 든
    ("던지", "든지"), ("든지", "던지"),
    ("하든", "하던"), ("하던", "하든"),
    ("갔든", "갔던"), ("갔던", "갔든"),
    ("먹든", "먹던"), ("먹던", "먹든"),
    ("있든", "있던"), ("있던", "있든"),
    ("했든", "했던"), ("했던", "했든"),
    ("보든", "보던"), ("보던", "보든"),
    # 데 / 대, 게 / 께
    ("는데", "는대"), ("는대", "는데"),
    ("던데", "던대"), ("던대", "던데"),
    ("든데", "던데"), ("던데", "든데"),
    ("은데", "은대"), ("은대", "은데"),
    ("ㄴ데", "ㄴ대"),
    ("할게", "할께"), ("할께", "할게"),
    ("갈게", "갈께"), ("갈께", "갈게"),
    ("볼게", "볼께"), ("볼께", "볼게"),
    ("줄게", "줄께"), ("줄께", "줄게"),
    ("올게", "올께"), ("올께", "올게"),
    # 로서 / 로써, 율 / 률
    ("로서", "로써"), ("로써", "로서"),
    # Frequent lexical / orthographic confusions
    ("며칠", "몇일"), ("몇일", "며칠"),
    ("금세", "금새"), ("금새", "금세"),
    ("오랜만", "오랫만"), ("오랫만", "오랜만"),
    ("어이없", "어의없"), ("어의없", "어이없"),
    ("설렘", "설레임"), ("설레임", "설렘"),
    ("역할", "역활"), ("역활", "역할"),
    ("왠지", "웬지"), ("웬지", "왠지"),
    ("웬만", "왠만"), ("왠만", "웬만"),
    ("어떻게", "어떡해"), ("어떡해", "어떻게"),
    ("맞히", "맞추"), ("맞추", "맞히"),
    ("맞춘", "맞힌"), ("맞힌", "맞춘"),
    ("맞춰", "맞혀"), ("맞혀", "맞춰"),
    ("맞췄", "맞혔"), ("맞혔", "맞췄"),
    ("가리키", "가르치"), ("가르치", "가리키"),
    ("가리켰", "가르쳤"), ("가르쳤", "가리켰"),
    ("부딪히", "부딪치"), ("부딪치", "부딪히"),
    ("반드시", "반듯이"), ("반듯이", "반드시"),
    ("늘이", "늘리"), ("늘리", "늘이"),
    ("붙이", "부치"), ("부치", "붙이"),
    ("이따가", "있다가"), ("있다가", "이따가"),
    ("다르", "틀리"), ("틀리", "다르"),
    ("다르다", "틀리다"), ("틀리다", "다르다"),
    ("다른", "틀린"), ("틀린", "다른"),
    ("낳", "낫"), ("낫", "낳"),
    ("빚", "빛"), ("빛", "빚"),
    ("바라", "바래"), ("바래", "바라"),
    ("건드리", "건들"), ("건들", "건드리"),
    ("깨끗이", "깨끗히"), ("깨끗히", "깨끗이"),
    ("곰곰이", "곰곰히"), ("곰곰히", "곰곰이"),
    ("일찍이", "일찍히"), ("일찍히", "일찍이"),
    ("솔직히", "솔직이"), ("솔직이", "솔직히"),
    ("어쨌든", "어쨋든"), ("어쨋든", "어쨌든"),
    ("실제로", "실재로"), ("실재로", "실제로"),
    ("결재", "결제"), ("결제", "결재"),
    ("지양", "지향"), ("지향", "지양"),
    ("수집", "수습"), ("수습", "수집"),
    ("무난", "문안"), ("문안", "무난"),
    ("곤란", "골란"), ("골란", "곤란"),
    ("일체", "일절"), ("일절", "일체"),
    ("너머", "넘어"), ("넘어", "너머"),
    ("예기", "얘기"), ("얘기", "예기"),
    ("베개", "배게"), ("배게", "베개"),
    ("무릎", "무릅"), ("무릅", "무릎"),
    ("희한", "희안"), ("희안", "희한"),
    ("가엽", "가엾"), ("가엾", "가엽"),
    ("서슴지", "서슴치"), ("서슴치", "서슴지"),
    ("통틀", "통털"), ("통털", "통틀"),
    ("뒤처", "뒤쳐"), ("뒤쳐", "뒤처"),
    ("쳐다", "처다"), ("처다", "쳐다"),
    ("치르", "치루"), ("치루", "치르"),
    ("잠그", "잠구"), ("잠구", "잠그"),
    ("담그", "담구"), ("담구", "담그"),
    ("들르", "들리"), ("들리", "들르"),
    ("봬요", "뵈요"), ("뵈요", "봬요"),
    ("뵀", "뵜"), ("뵜", "뵀"),
    ("있어", "이써"), ("이써", "있어"),
    ("없어", "업서"), ("업서", "없어"),
    ("좋아", "조아"), ("조아", "좋아"),
    ("개수", "갯수"), ("갯수", "개수"),
    ("대수", "댓수"), ("댓수", "대수"),
    ("횟수", "회수"), ("회수", "횟수"),
    ("초점", "촛점"), ("촛점", "초점"),
    ("대가", "댓가"), ("댓가", "대가"),
    ("숫자", "수자"),
    ("삼가", "삼가하"), ("삼가하", "삼가"),
    # Inflected / speech confusables added in spell-v2
    ("거예요", "거에요"), ("거에요", "거예요"),
    ("가르키", "가리키"), ("가리키", "가르키"),
    ("가르켜", "가리켜"), ("가리켜", "가르켜"),
    ("가르켰", "가리켰"), ("가리켰", "가르켰"),
    ("늘렸", "늘였"), ("늘였", "늘렸"),
    ("늘려", "늘여"), ("늘여", "늘려"),
    ("부쳤", "붙였"), ("붙였", "부쳤"),
    ("부쳐", "붙여"), ("붙여", "부쳐"),
    ("부딪혔", "부딪쳤"), ("부딪쳤", "부딪혔"),
    ("부딪혀", "부딪쳐"), ("부딪쳐", "부딪혀"),
    ("치렀", "치뤘"), ("치뤘", "치렀"),
    ("치러", "치뤄"), ("치뤄", "치러"),
    ("잠가", "잠궈"), ("잠궈", "잠가"),
    ("잠갔", "잠궜"), ("잠궜", "잠갔"),
    ("담가", "담궈"), ("담궈", "담가"),
    ("담갔", "담궜"), ("담궜", "담갔"),
    ("들러", "들려"), ("들려", "들러"),
    ("들렀", "들렸"), ("들렸", "들렀"),
    ("바라요", "바래요"), ("바래요", "바라요"),
    ("오래간만", "오랫간만"), ("오랫간만", "오래간만"),
]

# Post-position (josa) pairs: (with_batchim, without_batchim)
JOSA_PAIRS: list[tuple[str, str]] = [
    ("이", "가"),
    ("은", "는"),
    ("을", "를"),
    ("과", "와"),
    ("아", "야"),
    ("이랑", "랑"),
    ("이에요", "예요"),
    ("이라고", "라고"),
    ("이다", "다"),
    ("이라", "라"),
    ("이나", "나"),
    ("이든", "든"),
    ("이든지", "든지"),
]

# Common suffixes for stem+suffix decomposition search (longest first)
JOSA_EOMI_SUFFIXES: tuple[str, ...] = tuple(sorted({
    "으로", "로", "에서", "에게", "한테", "까지", "부터", "처럼", "보다", "마저", "조차", "밖에",
    "이나", "이든", "이랑", "이다", "입니다", "입니까", "이고", "이며", "이면", "이라", "이라고", "이에요", "예요",
    "은", "는", "이", "가", "을", "를", "에", "의", "와", "과", "도", "만", "랑",
    "하다", "한다", "했다", "하고", "해서", "하는", "하면", "하니", "하여", "하지", "하겠", "합니다", "했다가",
    "된다", "됐다", "되고", "돼서", "되는", "되면", "되지", "됩니다",
    "이다", "였다", "이었다", "이며", "이고", "이나", "지만", "는데", "으나", "면서", "으며", "어서", "아서", "해서",
    "어요", "아요",
    "했다", "였다", "었다", "았다", "겠다", "는다", "ㄴ다",
}, key=lambda s: (-len(s), s)))

FUZZY_SUFFIXES: tuple[str, ...] = tuple(sorted(
    set(JOSA_EOMI_SUFFIXES) | {
        "고", "며", "면", "니", "지", "다", "요", "서", "어", "아", "네", "게", "께",
        "들", "들이", "들의", "들을", "들은", "들에", "수록", "라고", "라는", "라며",
    },
    key=lambda s: (-len(s), s),
))

INNER_PUNCT_RE = re.compile(r'[()\[\]《》「」『』"\'“”‘’·,/~!?]')
_STEM_TAIL_HANGUL_RE = re.compile(r"^[0-9A-Za-z]+([가-힣]+)$")
_ASCII_ALNUM_RE = re.compile(r"[0-9A-Za-z]+")
_FUZZY_SUFFIX_SET = frozenset(FUZZY_SUFFIXES)
_PURE_BATCHIM_WITH = frozenset(("이", "은", "을", "과"))
_PURE_BATCHIM_WITHOUT = frozenset(("가", "는", "를", "와"))


def _batchim_compatible(stem_last: str, suf: str) -> bool:
    if not is_hangul_syllable(stem_last):
        return True
    has_b = has_final_consonant(stem_last)
    jong = jongsung_of(stem_last)
    if suf == "로":
        return (not has_b) or (jong == "ㄹ")
    if suf == "으로":
        return has_b and (jong != "ㄹ")
    if suf in _PURE_BATCHIM_WITH:
        return has_b
    if suf in _PURE_BATCHIM_WITHOUT:
        return not has_b
    return True

LEAD_PUNCT_RE = re.compile(r'^[\s"\'“”‘’(\[{<《「『【〈·….,:;!?~-]+')
TRAIL_PUNCT_RE = re.compile(r'[\s"\'“”‘’)\]}>》」』】〉·….,:;!?~-]+$')


def split_punct(word: str) -> tuple[str, str, str]:
    lead = ""
    trail = ""
    m = LEAD_PUNCT_RE.match(word)
    if m:
        lead = m.group(0)
        word = word[len(lead):]
    m2 = TRAIL_PUNCT_RE.search(word)
    if m2:
        trail = m2.group(0)
        word = word[:len(word) - len(trail)]
    return lead, word, trail


def fix_josa_candidates(core: str) -> list[str]:
    """Generate batchim-harmonized postposition candidates for `core`."""
    if len(core) < 2 or not all(is_hangul_syllable(c) for c in core[-2:]):
        return []
    out: list[str] = []
    # 1) Special case: (으)로 — ㄹ batchim or vowel takes 로, other batchim takes 으로
    if len(core) >= 3 and core.endswith("으로"):
        stem = core[:-2]
        prev = stem[-1]
        if not has_final_consonant(prev) or jongsung_of(prev) == "ㄹ":
            out.append(stem + "로")
    elif core.endswith("로") and len(core) >= 2:
        stem = core[:-1]
        prev = stem[-1]
        if has_final_consonant(prev) and jongsung_of(prev) != "ㄹ":
            out.append(stem + "으로")

    # 2) Standard batchim/non-batchim pairs (longer suffixes first)
    for with_b, without_b in sorted(JOSA_PAIRS, key=lambda p: -max(len(p[0]), len(p[1]))):
        if len(core) > len(with_b) and core.endswith(with_b):
            stem = core[:-len(with_b)]
            if not has_final_consonant(stem[-1]):
                out.append(stem + without_b)
        if len(core) > len(without_b) and core.endswith(without_b):
            stem = core[:-len(without_b)]
            if has_final_consonant(stem[-1]):
                out.append(stem + with_b)
    return list(dict.fromkeys(c for c in out if c != core))


def confusable_candidates(core: str) -> list[str]:
    """Generate Korean confusable-rule candidates for `core`."""
    if not HANGUL_SYLLABLE.search(core):
        return []
    out: list[str] = []
    for src, dst in CONFUSABLE_SUBSTRING_RULES:
        if src in core:
            cand = core.replace(src, dst)
            if cand != core:
                out.append(cand)
    # 율/률 rule: after vowel or ㄴ batchim -> 율, otherwise -> 률 (offer alternate if matched)
    for i, ch in enumerate(core):
        if ch in ("율", "률") and i > 0 and is_hangul_syllable(core[i - 1]):
            prev_jong = jongsung_of(core[i - 1])
            expected = "율" if prev_jong in ("", "ㄴ") else "률"
            if ch != expected:
                out.append(core[:i] + expected + core[i + 1:])
    return list(dict.fromkeys(out))


def qwerty_to_hangul_jamo(text: str) -> str:
    """Convert ASCII QWERTY letters in `text` to 2-beolsik jamo (decomposing Hangul syllables too)."""
    out: list[str] = []
    for ch in text:
        mapped = QWERTY_TO_JAMO.get(ch)
        if mapped is not None:
            out.append(mapped)
        else:
            out.append(decompose(ch))
    return "".join(out)


def qwerty_to_hangul(text: str) -> str:
    """Convert ASCII QWERTY letters in `text` into composed Hangul syllables via 2-beolsik mapping."""
    return compose(qwerty_to_hangul_jamo(text))


def jamo_edit_distance(a: str, b: str) -> float:
    """Weighted Damerau-Levenshtein distance over decomposed jamo or ASCII strings.

    Uses asymmetric 2-beolsik / QWERTY physical distances and empirical phonetic pairs
    via `_SUB_COST` (Jeon, Huang & Rim, 2010), plus reduced cost (0.55) for silent initial
    `'ㅇ'` insertion/deletion at syllable-boundary liaison (연철/분철), 1.00 ins/del, 0.90 transposition.
    """
    if a == b:
        return 0.0
    la, lb = len(a), len(b)
    if abs(la - lb) > 4:
        return float(max(la, lb))
    a_del_costs = [
        0.55 if (a[i] == "ㅇ" and i > 0 and a[i - 1] in CONSONANTS and i + 1 < la and a[i + 1] in VOWELS) else 1.0
        for i in range(la)
    ]
    b_ins_costs = [
        0.55 if (b[j] == "ㅇ" and j > 0 and b[j - 1] in CONSONANTS and j + 1 < lb and b[j + 1] in VOWELS) else 1.0
        for j in range(lb)
    ]
    sub_get = _SUB_COST.get
    dp = [[0.0] * (lb + 1) for _ in range(la + 1)]
    for i in range(1, la + 1):
        dp[i][0] = dp[i - 1][0] + a_del_costs[i - 1]
    for j in range(1, lb + 1):
        dp[0][j] = dp[0][j - 1] + b_ins_costs[j - 1]
    for i in range(1, la + 1):
        ca = a[i - 1]
        del_cost = a_del_costs[i - 1]
        for j in range(1, lb + 1):
            cb = b[j - 1]
            sub_cost = 0.0 if ca == cb else sub_get((ca, cb), 1.00)
            best = min(
                dp[i - 1][j] + del_cost,
                dp[i][j - 1] + b_ins_costs[j - 1],
                dp[i - 1][j - 1] + sub_cost,
            )
            if i > 1 and j > 1 and ca == b[j - 2] and a[i - 2] == cb:
                trans = dp[i - 2][j - 2] + 0.90
                if trans < best:
                    best = trans
            dp[i][j] = best
    return dp[la][lb]


def _deletes_1(s: str) -> list[str]:
    return [s[:i] + s[i + 1:] for i in range(len(s))]


def _deletes_2(s: str) -> set[str]:
    out: set[str] = set()
    n = len(s)
    for i in range(n):
        s1 = s[:i] + s[i + 1:]
        for j in range(i, n - 1):
            out.add(s1[:j] + s1[j + 1:])
    return out


class SpellCandidateGenerator:
    """Fast SymSpell-jamo + confusable + josa + QWERTY candidate generator over train vocabularies."""

    def __init__(
        self,
        word_freq: dict[str, int],
        stem_freq: dict[str, int] | None = None,
        eng_freq: dict[str, int] | None = None,
    ):
        self.word_freq = word_freq
        self.stem_freq = stem_freq or {}
        self.eng_freq: dict[str, int] = dict(eng_freq or {})
        for w, freq in word_freq.items():
            if w.isascii() and PURE_ASCII_ALPHA.fullmatch(w) and w not in self.eng_freq:
                self.eng_freq[w] = freq

        # Build jamo lookup & delete-1 index for full words
        self._words: list[str] = list(word_freq.keys())
        self._word_jamos: list[str] = [decompose(w) for w in self._words]
        self._word_jamo_map: dict[str, list[int]] = defaultdict(list)
        self._word_del1: dict[str, list[int]] = defaultdict(list)
        for wid, j in enumerate(self._word_jamos):
            self._word_jamo_map[j].append(wid)
            if len(j) >= 2:
                for d in set(_deletes_1(j)):
                    self._word_del1[d].append(wid)

        # Build jamo lookup & delete-1 index for stems
        self._stems: list[str] = list(self.stem_freq.keys())
        self._stem_jamos: list[str] = [decompose(s) for s in self._stems]
        self._stem_jamo_map: dict[str, list[int]] = defaultdict(list)
        self._stem_del1: dict[str, list[int]] = defaultdict(list)
        for sid, j in enumerate(self._stem_jamos):
            self._stem_jamo_map[j].append(sid)
            if len(j) >= 2:
                for d in set(_deletes_1(j)):
                    self._stem_del1[d].append(sid)

        # Optional lowercase SymSpell delete-1 index for English / alphanumeric QWERTY words
        self._eng_words: list[str] = list(self.eng_freq.keys())
        self._eng_map: dict[str, list[int]] = defaultdict(list)
        self._eng_del1: dict[str, list[int]] = defaultdict(list)
        for eid, ew in enumerate(self._eng_words):
            el = ew.lower()
            self._eng_map[el].append(eid)
            if len(el) >= 2:
                for d in set(_deletes_1(el)):
                    self._eng_del1[d].append(eid)

        self._suffix_jamos: list[tuple[str, str]] = [(s, decompose(s)) for s in FUZZY_SUFFIXES]

    @classmethod
    def from_file(cls, path: str | Path) -> "SpellCandidateGenerator":
        path = Path(path)
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8") as handle:
            data = json.load(handle)
        return cls(data["word_freq"], data.get("stem_freq", {}), data.get("eng_freq"))

    def _lookup_jamo(self, q_jamo: str, jamo_map: dict[str, list[int]], del1_map: dict[str, list[int]],
                     max_hits: int = 80) -> set[int]:
        """Find vocabulary IDs within jamo edit distance <= ~2 (including neighbor 1-sub + 1-edit)."""
        hits: set[int] = set()
        # Exact jamo match (handles rare Unicode/decomposition collisions)
        if q_jamo in jamo_map:
            hits.update(jamo_map[q_jamo])
        # Delete-1 covers 1-ins, 1-del, 1-sub, 1-transpose
        if q_jamo in del1_map:
            hits.update(del1_map[q_jamo])
        q_del1 = set(_deletes_1(q_jamo))
        for d in q_del1:
            if d in jamo_map:
                hits.update(jamo_map[d])
            if d in del1_map:
                hits.update(del1_map[d])
        # If still sparse or for 2-edit keyboard/deletion recall, query delete-2 + 1-neighbor-sub delete-1
        if len(q_jamo) >= 3:
            for d2 in _deletes_2(q_jamo):
                if d2 in jamo_map:
                    hits.update(jamo_map[d2])
                if len(hits) < max_hits and d2 in del1_map:
                    hits.update(del1_map[d2][:12])
        # Query 1-keyboard/phonetic substitution variants through del1_map to catch 2-substitution typos
        if len(hits) < max_hits and len(q_jamo) >= 3:
            for i, ch in enumerate(q_jamo):
                for alt in _LOOKUP_SUB_ALTS.get(ch, ()):
                    mut = q_jamo[:i] + alt + q_jamo[i + 1:]
                    if mut in jamo_map:
                        hits.update(jamo_map[mut])
                    if mut in del1_map:
                        hits.update(del1_map[mut][:8])
                    for d in _deletes_1(mut):
                        if d in jamo_map:
                            hits.update(jamo_map[d])
                        if len(hits) < max_hits and d in del1_map:
                            hits.update(del1_map[d][:6])
        return hits

    def _compound_split_candidates(self, word: str, max_per_side: int = 4) -> list[tuple[str, int]]:
        """Recover 2-part OOV compound nouns when one side is known frequent and the other has <=1.65 edit."""
        if not PURE_HANGUL_WORD.fullmatch(word) or len(word) < 4 or word in self.word_freq:
            return []

        def side_fixes(part: str, *, allow_josa: bool) -> list[tuple[str, int]]:
            p_jamo = decompose(part)
            scored_side: dict[str, tuple[float, int]] = {}

            def add_side(cand_part: str, cfreq: int, cand_jamo: str | None = None) -> None:
                if not cand_part or cand_part == part or cfreq < 5:
                    return
                cj = cand_jamo if cand_jamo is not None else decompose(cand_part)
                ed = jamo_edit_distance(p_jamo, cj)
                if not (0.0 < ed <= 1.65):
                    return
                score = ed - 0.05 * math.log1p(cfreq)
                prev = scored_side.get(cand_part)
                if prev is None or score < prev[0] or (score == prev[0] and cfreq > prev[1]):
                    scored_side[cand_part] = (score, cfreq)

            for wid in self._lookup_jamo(p_jamo, self._word_jamo_map, self._word_del1, max_hits=25):
                cw = self._words[wid]
                add_side(cw, max(self.word_freq[cw], self.stem_freq.get(cw, 0)), self._word_jamos[wid])
            for sid in self._lookup_jamo(p_jamo, self._stem_jamo_map, self._stem_del1, max_hits=15):
                cs = self._stems[sid]
                add_side(cs, max(self.word_freq.get(cs, 0), self.stem_freq[cs]), self._stem_jamos[sid])
            if allow_josa:
                for cj_cand in fix_josa_candidates(part):
                    cfreq = max(self.word_freq.get(cj_cand, 0), self.stem_freq.get(cj_cand, 0))
                    add_side(cj_cand, cfreq)
            for cf_cand in confusable_candidates(part):
                cfreq = max(self.word_freq.get(cf_cand, 0), self.stem_freq.get(cf_cand, 0))
                add_side(cf_cand, cfreq)

            ranked_side = sorted(scored_side.items(), key=lambda kv: (kv[1][0], -kv[1][1], kv[0]))
            return [(cand_p, info[1]) for cand_p, info in ranked_side[:max_per_side]]

        best: dict[str, int] = {}
        for cut in range(2, len(word) - 1):
            left = word[:cut]
            right = word[cut:]
            lfreq = max(self.word_freq.get(left, 0), self.stem_freq.get(left, 0))
            rfreq = max(self.word_freq.get(right, 0), self.stem_freq.get(right, 0))
            if lfreq >= 10:
                for cand_r, cr_freq in side_fixes(right, allow_josa=True):
                    cand = left + cand_r
                    if cand != word:
                        mfreq = min(lfreq, cr_freq)
                        if mfreq > best.get(cand, -1):
                            best[cand] = mfreq
            if rfreq >= 10:
                for cand_l, cl_freq in side_fixes(left, allow_josa=False):
                    cand = cand_l + right
                    if cand != word:
                        mfreq = min(cl_freq, rfreq)
                        if mfreq > best.get(cand, -1):
                            best[cand] = mfreq
        return list(best.items())

    def generate(self, word: str, max_candidates: int = 15, include_keep: bool = True) -> list[str]:
        """Return ranked correction candidates for a single whitespace-separated `word`."""
        lead, core, trail = split_punct(word)
        if not core:
            return [word] if include_keep else []
        is_ascii_alpha = bool(PURE_ASCII_ALPHA.fullmatch(core))
        has_hangul_syll = bool(HANGUL_SYLLABLE.search(core))
        has_jamo = bool(HANGUL_JAMO_RE.search(core))
        has_ascii_alpha = bool(ASCII_ALPHA_RE.search(core))
        if not has_hangul_syll and not has_jamo and not (is_ascii_alpha and len(core) >= 3):
            return [word] if include_keep else []

        q_jamo = decompose(core)
        # Map candidate_core -> (rule_bonus, stem_penalty, freq, base_jamo)
        scored: dict[str, tuple[float, float, int, str]] = {}

        def register(
            cand_core: str,
            rule_bonus: float = 0.0,
            stem_penalty: float = 0.0,
            freq_override: int | None = None,
            base_jamo: str = q_jamo,
        ) -> None:
            if not cand_core:
                return
            freq = freq_override if freq_override is not None else self.word_freq.get(cand_core, self.stem_freq.get(cand_core, 1))
            prev = scored.get(cand_core)
            if prev is None or rule_bonus > prev[0] or (rule_bonus == prev[0] and freq > prev[2]):
                scored[cand_core] = (rule_bonus, stem_penalty, freq, base_jamo)

        batchim_particles = ("이", "가", "은", "는", "을", "를", "과", "와", "으로", "로")

        def register_stem_hit(
            cand_stem: str, stem_q: str, suf: str, stem_penalty: float, freq: int, base_jamo: str
        ) -> None:
            if not cand_stem or cand_stem == stem_q:
                return
            combined = cand_stem + suf
            register(combined, rule_bonus=0.0, stem_penalty=stem_penalty, freq_override=freq, base_jamo=base_jamo)
            if suf in batchim_particles and is_hangul_syllable(cand_stem[-1]) and is_hangul_syllable(stem_q[-1]):
                batchim_diff = has_final_consonant(cand_stem[-1]) != has_final_consonant(stem_q[-1])
                rieul_diff = suf in ("으로", "로") and jongsung_of(cand_stem[-1]) != jongsung_of(stem_q[-1])
                if batchim_diff or rieul_diff or (not _batchim_compatible(cand_stem[-1], suf)):
                    for fixed in fix_josa_candidates(combined):
                        register(
                            fixed, rule_bonus=0.1, stem_penalty=stem_penalty, freq_override=freq, base_jamo=base_jamo
                        )

        def search_hangul_core(target_core: str, base_jamo: str, *, allow_stem_split: bool = True) -> None:
            target_jamo = decompose(target_core)
            for c in fix_josa_candidates(target_core):
                if (
                    target_core.endswith("로")
                    and c == target_core[:-1] + "으로"
                    and self.word_freq.get(target_core, 0) >= 5
                    and self.word_freq.get(c, 0) < self.word_freq.get(target_core, 0)
                ):
                    continue
                if (
                    target_core.endswith("다")
                    and c == target_core[:-1] + "이다"
                    and c not in self.word_freq
                ):
                    stem_cand = target_core[:-1]
                    if (
                        max(self.word_freq.get(stem_cand, 0), self.stem_freq.get(stem_cand, 0)) < 5
                        or stem_cand[-1] in "했었았겠였는하되있없"
                    ):
                        continue
                register(c, rule_bonus=0.65, freq_override=max(self.word_freq.get(c, 0), 20), base_jamo=base_jamo)
            for c in confusable_candidates(target_core):
                register(c, rule_bonus=0.55, freq_override=max(self.word_freq.get(c, 0), 15), base_jamo=base_jamo)

            wids = self._lookup_jamo(target_jamo, self._word_jamo_map, self._word_del1)
            for wid in wids:
                cand = self._words[wid]
                if cand != core:
                    register(cand, rule_bonus=0.0, stem_penalty=0.0, freq_override=self.word_freq[cand], base_jamo=base_jamo)

            if not allow_stem_split:
                return
            for suf in JOSA_EOMI_SUFFIXES:
                if len(target_core) > len(suf) + 1 and target_core.endswith(suf):
                    stem_q = target_core[:-len(suf)]
                    if not HANGUL_SYLLABLE.search(stem_q):
                        continue
                    stem_q_jamo = decompose(stem_q)
                    if len(stem_q_jamo) < 2:
                        continue
                    s_hits = self._lookup_jamo(stem_q_jamo, self._stem_jamo_map, self._stem_del1, max_hits=30)
                    for sid in s_hits:
                        cand_stem = self._stems[sid]
                        register_stem_hit(cand_stem, stem_q, suf, 0.25, self.stem_freq[cand_stem], base_jamo)
                    if stem_q not in self.stem_freq:
                        w_stem_hits = self._lookup_jamo(stem_q_jamo, self._word_jamo_map, self._word_del1, max_hits=20)
                        for wid in w_stem_hits:
                            cand_stem = self._words[wid]
                            register_stem_hit(cand_stem, stem_q, suf, 0.35, self.word_freq[cand_stem], base_jamo)

        def register_derived_hangul(converted: str, derived_jamo: str, *, allow_near_complete: bool = False) -> None:
            if not converted or converted == core or ASCII_ALPHA_RE.search(converted):
                return
            is_clean_hangul = bool(PURE_HANGUL_WORD.fullmatch(converted)) and len(converted) >= 2
            syll_n = len(HANGUL_SYLLABLE.findall(converted))
            jamo_n = len(HANGUL_JAMO_RE.findall(converted))
            is_near = allow_near_complete and syll_n >= 2 and jamo_n <= 1
            if not (is_clean_hangul or is_near):
                return
            if is_clean_hangul and converted in self.word_freq:
                register(
                    converted,
                    rule_bonus=0.60,
                    freq_override=self.word_freq[converted],
                    base_jamo=derived_jamo,
                )
            search_hangul_core(converted, base_jamo=derived_jamo, allow_stem_split=is_clean_hangul)

        # 0. Recover 2-beolsik Hangul from English-mode / mixed-QWERTY typing & standalone jamo
        if is_ascii_alpha and len(core) >= 3:
            raw_q_jamo = qwerty_to_hangul_jamo(core)
            register_derived_hangul(compose(raw_q_jamo), raw_q_jamo, allow_near_complete=True)
            if self._eng_map:
                for eid in self._lookup_jamo(core.lower(), self._eng_map, self._eng_del1, max_hits=40):
                    cand = self._eng_words[eid]
                    if cand != core:
                        register(
                            cand,
                            rule_bonus=0.0,
                            stem_penalty=0.0,
                            freq_override=self.eng_freq[cand],
                            base_jamo=core,
                        )
        elif has_ascii_alpha and has_hangul_syll:
            mixed_jamo = qwerty_to_hangul_jamo(core)
            register_derived_hangul(compose(mixed_jamo), mixed_jamo, allow_near_complete=False)

        if has_jamo:
            register_derived_hangul(compose(q_jamo), q_jamo, allow_near_complete=False)

        # 1~3. Standard Hangul rules, SymSpell whole-word, and stem+suffix search
        if has_hangul_syll or has_jamo:
            search_hangul_core(core, base_jamo=q_jamo, allow_stem_split=True)

        # 4. Known exact / alphanumeric stem + fuzzy suffix typo recovery
        if has_hangul_syll and len(core) >= 2:
            for cut in range(max(1, len(core) - 3), len(core)):
                stem = core[:cut]
                tail = core[cut:]
                stem_freq_val = self.word_freq.get(stem, 0) or self.stem_freq.get(stem, 0)
                is_ascii_alnum_stem = False
                if not stem_freq_val:
                    m_alpha = _STEM_TAIL_HANGUL_RE.match(stem)
                    if m_alpha:
                        hangul_part = m_alpha.group(1)
                        stem_freq_val = self.word_freq.get(hangul_part, 0) or self.stem_freq.get(hangul_part, 0)
                    elif (
                        len(stem) >= 2
                        and _ASCII_ALNUM_RE.fullmatch(stem)
                        and len(tail) <= 2
                        and PURE_HANGUL_WORD.fullmatch(tail)
                        and tail not in _FUZZY_SUFFIX_SET
                    ):
                        stem_freq_val = 20
                        is_ascii_alnum_stem = True
                if not stem_freq_val:
                    continue
                tail_j = decompose(tail)
                tl = len(tail_j)
                stem_last = stem[-1]
                for suf, suf_j in self._suffix_jamos:
                    if abs(tl - len(suf_j)) > 2 or suf == tail:
                        continue
                    if not _batchim_compatible(stem_last, suf):
                        continue
                    ed = jamo_edit_distance(tail_j, suf_j)
                    max_suf_ed = 1.15 if is_ascii_alnum_stem else 1.65
                    if not (0.0 < ed <= max_suf_ed):
                        continue
                    combined = stem + suf
                    in_wf = self.word_freq.get(combined, 0)
                    if in_wf > 0:
                        register(combined, rule_bonus=0.0, stem_penalty=0.10, freq_override=in_wf)
                    elif is_ascii_alnum_stem:
                        register(combined, rule_bonus=0.0, stem_penalty=0.35, freq_override=20)
                    elif (len(suf) >= 2 and ed <= 1.65) or (len(suf) == 1 and ed <= 1.15):
                        register(combined, rule_bonus=0.0, stem_penalty=0.85, freq_override=min(stem_freq_val, 30))

        # 5. Inner-punctuation words: recover trailing suffix or misspelled Hangul segments
        if has_hangul_syll and INNER_PUNCT_RE.search(core):
            parts = re.split(r"([가-힣]+)", core)
            hangul_pis = [pi for pi, seg in enumerate(parts) if seg and PURE_HANGUL_WORD.fullmatch(seg)]
            if hangul_pis:
                last_pi = hangul_pis[-1]
                for pi in hangul_pis:
                    seg = parts[pi]
                    seg_j = decompose(seg)
                    if pi == last_pi:
                        prev_idx = sum(len(x) for x in parts[:pi]) - 1
                        prev_ch = core[prev_idx] if prev_idx >= 0 else ""
                        sl = len(seg_j)
                        for suf, suf_j in self._suffix_jamos:
                            if abs(sl - len(suf_j)) > 2 or suf == seg:
                                continue
                            if not _batchim_compatible(prev_ch, suf):
                                continue
                            if 0.0 < jamo_edit_distance(seg_j, suf_j) <= 1.65:
                                parts_copy = list(parts)
                                parts_copy[pi] = suf
                                register("".join(parts_copy), rule_bonus=0.0, stem_penalty=0.25, freq_override=50)
                    if len(seg) >= 2 and (pi == last_pi or seg not in self.word_freq):
                        seg_cands: list[tuple[float, str]] = [(0.5, c) for c in confusable_candidates(seg)]
                        for wid in self._lookup_jamo(seg_j, self._word_jamo_map, self._word_del1, max_hits=30):
                            w = self._words[wid]
                            if w == seg:
                                continue
                            ed = jamo_edit_distance(seg_j, self._word_jamos[wid])
                            if ed <= 2.4:
                                seg_cands.append((ed - 0.05 * math.log1p(self.word_freq[w]), w))
                        if len(seg) >= 4 and seg not in self.word_freq:
                            for cw, cfreq in self._compound_split_candidates(seg, max_per_side=3):
                                ed = jamo_edit_distance(seg_j, decompose(cw))
                                seg_cands.append((ed + 0.30 - 0.05 * math.log1p(cfreq), cw))
                        seg_cands.sort(key=lambda x: (x[0], x[1]))
                        seen_seg: set[str] = set()
                        for _, cand_seg in seg_cands:
                            if cand_seg in seen_seg:
                                continue
                            seen_seg.add(cand_seg)
                            parts_copy = list(parts)
                            parts_copy[pi] = cand_seg
                            register(
                                "".join(parts_copy),
                                rule_bonus=0.0,
                                stem_penalty=0.25,
                                freq_override=max(self.word_freq.get(cand_seg, 0), 20),
                            )
                            if len(seen_seg) >= 6:
                                break

        # 6. Two-part OOV compound noun split recovery (Jeon, Huang & Rim, 2010 Sec. 5.3)
        if PURE_HANGUL_WORD.fullmatch(core) and len(core) >= 4 and core not in self.word_freq:
            for cand, cfreq in self._compound_split_candidates(core):
                register(cand, rule_bonus=0.0, stem_penalty=0.30, freq_override=cfreq)

        ranked: list[tuple[float, str]] = []
        for cand_core, (bonus, spen, freq, base_jamo) in scored.items():
            if cand_core == core:
                continue
            cand_jamo = cand_core if cand_core.isascii() else decompose(cand_core)
            ed = jamo_edit_distance(base_jamo, cand_jamo)
            # Filter wildly distant candidates unless rule-generated
            if bonus == 0.0 and ed > 2.6:
                continue
            rank_ed = min(ed, 1.0) if bonus >= 0.5 else ed
            cost = rank_ed + spen - bonus - 0.08 * math.log1p(freq)
            ranked.append((cost, cand_core))

        ranked.sort(key=lambda x: (x[0], x[1]))
        cores = [c for _, c in ranked[: (max_candidates - 1 if include_keep else max_candidates)]]
        result = [f"{lead}{c}{trail}" for c in cores]
        if include_keep and word not in result:
            result.insert(0, word) if not result else result.append(word)
        return result[:max_candidates]


def _self_test() -> None:
    assert decompose("한글") == "ㅎㅏㄴㄱㅡㄹ"
    assert decompose("값") == "ㄱㅏㅂㅅ"
    assert compose("ㅎㅏㄴㄱㅡㄹ") == "한글"
    assert compose("ㄱㅏㅂㅅ") == "값"
    assert has_final_consonant("책") and not has_final_consonant("사과")
    assert "사과를" in fix_josa_candidates("사과을")
    assert "서울로" in fix_josa_candidates("서울으로")
    assert "집으로" in fix_josa_candidates("집로")
    assert "돼요" in confusable_candidates("되요")
    assert qwerty_to_hangul("dkssudgktpdy") == "안녕하세요"
    assert qwerty_to_hangul("gksrmf") == "한글"
    assert qwerty_to_hangul("사rhk를") == "사과를"
    assert (
        keyboard_sub_cost("ㄱ", "ㄷ")
        < keyboard_sub_cost("ㄱ", "ㅇ")
        < keyboard_sub_cost("ㄱ", "ㅎ")
        < keyboard_sub_cost("ㄱ", "ㅣ")
    )
    assert keyboard_sub_cost("ㅅ", "ㅛ") == 0.60
    assert keyboard_sub_cost("e", "r") == 0.60 < keyboard_sub_cost("e", "m")
    # Asymmetric empirical phonetic costs (Jeon, Huang & Rim, 2010) & aspirated/tense/ㅎ-ㅇ pairs
    assert keyboard_sub_cost("ㅆ", "ㅅ") < keyboard_sub_cost("ㅅ", "ㅆ")
    assert keyboard_sub_cost("ㅐ", "ㅔ") < keyboard_sub_cost("ㅔ", "ㅐ")
    assert keyboard_sub_cost("ㅠ", "ㅜ") < keyboard_sub_cost("ㅜ", "ㅠ")
    assert keyboard_sub_cost("ㅍ", "ㅂ") == 0.50 < keyboard_sub_cost("ㅂ", "ㅍ")
    assert keyboard_sub_cost("ㅎ", "ㅇ") == 0.48 < keyboard_sub_cost("ㅇ", "ㅎ")
    assert keyboard_sub_cost("ㅋ", "ㄱ") == 0.54 < keyboard_sub_cost("ㄱ", "ㅋ")
    assert keyboard_sub_cost("ㅌ", "ㄷ") == 0.54 < keyboard_sub_cost("ㄷ", "ㅌ")
    # Silent 'ㅇ' liaison distance < general deletion/insertion distance (1.0)
    assert jamo_edit_distance(decompose("채그로"), decompose("책으로")) == 0.55
    assert jamo_edit_distance(decompose("엉이가"), decompose("어이가")) == 1.00

    mini_words = {
        "항상": 500, "항산": 20, "항시": 30, "과학": 400, "분야에서": 100, "먹었는데": 50, "돼요": 300,
        "한국어": 400, "사과를": 200, "테헤란로": 100, "경로": 500, "굴뚝새": 50, "이노센스": 50,
        "안녕하세요": 600, "한글": 300, "시작": 400, "양": 7807, "랙": 364, "둥": 124,
        "공식": 500, "인증": 500, "국가": 500, "부채": 500, "책으로": 300, "태그로": 200,
        "기록했다": 400, "학생": 200,
    }
    mini_stems = {"항상": 500, "분야": 300, "먹었": 200, "사과": 300, "도움": 300}
    mini_eng = {"PyTorch": 500, "Google": 400, "Python": 450}
    gen = SpellCandidateGenerator(mini_words, mini_stems, eng_freq=mini_eng)
    c1 = gen.generate("항샅", max_candidates=10)
    assert "항상" in c1 and "항샅" in c1, c1
    c2 = gen.generate("분아에서.", max_candidates=10)
    assert "분야에서." in c2 and "분아에서." in c2, c2
    c3 = gen.generate("되요", max_candidates=10)
    assert "돼요" in c3, c3
    c4 = gen.generate("한구거", max_candidates=10)
    assert "한국어" in c4, c4
    c5 = gen.generate("사광을", max_candidates=10)
    assert "사과를" in c5, c5
    c_ro1 = gen.generate("테헤란로", max_candidates=10)
    assert "테헤란으로" not in c_ro1, c_ro1
    c_ro2 = gen.generate("경로", max_candidates=10)
    assert "경으로" not in c_ro2, c_ro2
    c_josa = gen.generate("집로", max_candidates=10)
    assert "집으로" in c_josa, c_josa
    c_fs1 = gen.generate("굴뚝새늦", max_candidates=10)
    assert "굴뚝새는" in c_fs1, c_fs1
    c_fs2 = gen.generate("이노센스위", max_candidates=10)
    assert "이노센스의" in c_fs2, c_fs2
    c_ip1 = gen.generate("야인(野人)든이", max_candidates=10)
    assert "야인(野人)들이" in c_ip1, c_ip1
    c_ip2 = gen.generate("당대회()느", max_candidates=10)
    assert "당대회()는" in c_ip2, c_ip2
    c_cf1 = gen.generate("거에요", max_candidates=10)
    assert "거예요" in c_cf1, c_cf1
    c_cf2 = gen.generate("가르키고", max_candidates=10)
    assert "가리키고" in c_cf2, c_cf2
    # QWERTY <-> Hangul, standalone jamo, boundary typos, and English protection
    c_q1 = gen.generate("dkssudgktpdy", max_candidates=10)
    assert c_q1[0] == "안녕하세요", c_q1
    c_q2 = gen.generate("gksrmf", max_candidates=10)
    assert c_q2[0] == "한글", c_q2
    c_q3 = gen.generate("사rhk를", max_candidates=10)
    assert "사과를" in c_q3, c_q3
    c_j1 = gen.generate("ㅎㅏㄴㄱㅡㄹ", max_candidates=10)
    assert c_j1[0] == "한글", c_j1
    c_j2 = gen.generate("하ㄴ국어", max_candidates=10)
    assert "한국어" in c_j2, c_j2
    c_bd = gen.generate("ㅛㅣ작", max_candidates=10)
    assert "시작" in c_bd, c_bd
    for clean_eng in ("GPU", "pytest", "to", "did", "for", "end", "API", "GPU에서"):
        assert gen.generate(clean_eng, max_candidates=10) == [clean_eng], (clean_eng, gen.generate(clean_eng))
    assert "PyTorch" in gen.generate("Pytoch", max_candidates=10), gen.generate("Pytoch")
    assert "Google" in gen.generate("Gogle", max_candidates=10), gen.generate("Gogle")
    assert "Python" in gen.generate("pytohn", max_candidates=10), gen.generate("pytohn")
    # Liaison, 2-part OOV compounds, and DEV diagnostic fixes
    assert gen.generate("채그로", max_candidates=10)[0] == "책으로", gen.generate("채그로")
    assert "공식인증" in gen.generate("궁식인증", max_candidates=10), gen.generate("궁식인증")
    assert "국가부채" in gen.generate("국가부래", max_candidates=10), gen.generate("국가부래")
    c_rec = gen.generate("기롴했다.", max_candidates=10)
    assert "기록했다." in c_rec and "기롴했이다." not in c_rec, c_rec
    assert "학생이다" in gen.generate("학생다", max_candidates=10), gen.generate("학생다")
    assert "도움이" in gen.generate("고움가", max_candidates=10), gen.generate("고움가")
    assert "220PPI를" in gen.generate("220PPI럴", max_candidates=10), gen.generate("220PPI럴")
    print("spell_candidates self-test passed:", c1[:3], c_q1[:2], c_bd[:2], gen.generate("Pytoch", max_candidates=3))
