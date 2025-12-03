from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from pypinyin import pinyin, Style

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"] # for Chinese characters

# ---------- 1. Basic helpers ----------

def is_chinese_char(ch: str) -> bool:
    """Rudimentary check if a character is CJK Unified Ideograph."""
    return "\u4e00" <= ch <= "\u9fff"


def char_to_initial(ch: str) -> str | None:
    """
    Map a single Chinese character to the first letter of its pinyin.
    Example: '我' -> 'w', '要' -> 'y'.

    Returns None for non-Chinese chars or if pinyin cannot be obtained.
    """
    if not is_chinese_char(ch):
        return None

    py_list = pinyin(ch, style=Style.NORMAL, strict=False)
    if not py_list or not py_list[0]:
        return None

    syllable = py_list[0][0]  # e.g. "wo", "yao", "wan"
    return syllable[0].lower()


def build_initial_to_char_counts(texts: Iterable[str]) -> Dict[str, Counter]:
    """
    Build frequency table:
      initial_letter -> Counter({character: count, ...})
    from an iterable of Chinese texts.
    """
    mapping: Dict[str, Counter] = defaultdict(Counter)

    for text in texts:
        for ch in text:
            initial = char_to_initial(ch)
            if initial is None:
                continue
            mapping[initial][ch] += 1

    return mapping


# ---------- 2. Build matrix for selected characters & initials ----------

def build_matrix(
    initial_to_counts: Dict[str, Counter],
    initials: List[str] | None = None,
    extra_chars: List[str] | None = None,
    top_k_per_initial: int = 5,
) -> Tuple[List[str], List[str], np.ndarray]:
    """
    Build a matrix of shape (num_chars, num_initials), where
    entry [i, j] = frequency of character_i with initial_j in the corpus.

    - initials: which initials to keep (e.g. ["w", "y", "s", "j"]).
      If None, uses all initials found.
    - extra_chars: characters that must always be included as rows,
      even if they are not top-k.
    - top_k_per_initial: additionally include the top-k characters
      (by frequency) for each initial.
    """
    if initials is None:
        initials = sorted(initial_to_counts.keys())
    else:
        initials = sorted(set(initials) & set(initial_to_counts.keys()))

    # collect characters to appear as rows
    char_set = set(extra_chars or [])
    for init in initials:
        for ch, _ in initial_to_counts[init].most_common(top_k_per_initial):
            char_set.add(ch)

    chars = sorted(char_set)  # row order

    # build matrix
    mat = np.zeros((len(chars), len(initials)), dtype=int)
    for i, ch in enumerate(chars):
        for j, init in enumerate(initials):
            mat[i, j] = initial_to_counts[init][ch]

    return initials, chars, mat


# ---------- 3. Pretty-print as table ----------

def print_matrix(initials: List[str], chars: List[str], mat: np.ndarray) -> None:
    """
    Print matrix in a text format like:

           w   y   s   j
    我    34   0   0   0
    要     0  29   0   0
    """
    # Header
    header = "    " + " ".join(f"{init:>4}" for init in initials)
    print(header)

    # Rows
    for i, ch in enumerate(chars):
        row_counts = " ".join(f"{mat[i, j]:>4}" for j in range(len(initials)))
        print(f"{ch:<2}  {row_counts}")


# ---------- 4. Optional: visualize as heatmap ----------

def show_heatmap(initials: List[str], chars: List[str], mat: np.ndarray) -> None:
    """
    Simple heatmap visualization.
    """
    plt.figure(figsize=(8, max(4, len(chars) * 0.4)))
    plt.imshow(mat, aspect="auto")
    plt.colorbar(label="Frequency")
    plt.xticks(range(len(initials)), initials)
    plt.yticks(range(len(chars)), chars)
    plt.xlabel("Pinyin initial")
    plt.ylabel("Chinese character")
    plt.title("Character frequency conditioned on pinyin initial")
    plt.tight_layout()
    plt.show()


# ---------- 5. Example usage ----------

if __name__ == "__main__":
    # Example training data.
    # Replace this with your own corpus (webtoons, BabyLM, etc.).
    training_texts = [
        "我要睡觉了。",
        "我喜欢中国的网络漫画。",
        "我想睡觉，但是我还要学习。",
        "要努力学习，也要好好休息。",
        "我玩游戏，不想写作业。",
        "他在玩完游戏以后就去睡觉。",
        "袜子和五本书都在我的房间里。",
    ]

    # 1) Build counts
    initial_to_counts = build_initial_to_char_counts(training_texts)

    # 2) Define initials and characters of interest
    initials_of_interest = ["w", "y", "s", "j"]  # extend if you like
    extra_chars = ["我", "玩", "完", "袜", "五", "要", "是", "觉"]

    # 3) Build matrix (top 5 chars per initial + extra_chars)
    initials, chars, mat = build_matrix(
        initial_to_counts,
        initials=initials_of_interest,
        extra_chars=extra_chars,
        top_k_per_initial=5,
    )

    # 4) Print table in the desired format
    print_matrix(initials, chars, mat)

    # 5) Show heatmap (optional)
    show_heatmap(initials, chars, mat)
