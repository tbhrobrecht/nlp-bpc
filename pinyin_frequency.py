from __future__ import annotations
from collections import Counter, defaultdict
from typing import Dict, Iterable
from pypinyin import pinyin, Style

import pandas as pd

dataframe = pd.read_csv("C:\\Users\\tbhro\\PycharmProjects\\nlp-bpc\\zho_data\\sample_text.csv", encoding="utf-8")
zho_text = dataframe["text"].tolist()
# print(zho_text)
# Remove \r\n from each string in the zho_text list
zho_text_cleaned = [text.replace("\r\n", "") for text in zho_text]
print(zho_text_cleaned)


# train
def is_chinese_char(ch: str) -> bool:
    """Rudimentary check if ch is a CJK Unified Ideograph."""
    return "\u4e00" <= ch <= "\u9fff"


def char_to_initial(ch: str) -> str | None:
    """
    Convert a single Chinese character to the first letter of its pinyin.
    Example: '我' -> 'w', '要' -> 'y'
    Returns None if pinyin can't be obtained.
    """
    if not is_chinese_char(ch):
        return None

    py_list = pinyin(ch, style=Style.NORMAL, strict=False)
    if not py_list or not py_list[0]:
        return None

    syllable = py_list[0][0]  # e.g. 'wo', 'yao', 'shui'
    return syllable[0].lower()  # first letter: 'w', 'y', 's', 'j', ...


def build_initial_to_char_counts(texts: Iterable[str]) -> Dict[str, Counter]:
    """
    Build a frequency table:
    initial_letter -> Counter({character: count, ...})
    from an iterable of Chinese texts.
    """
    mapping: Dict[str, Counter] = defaultdict(Counter)

    for text in texts:
        for ch in text:
            if not is_chinese_char(ch):
                continue
            initial = char_to_initial(ch)
            if initial is None:
                continue
            mapping[initial][ch] += 1

    return mapping


# decoding  
def build_initial_to_best_char(initial_to_counts: Dict[str, Counter]) -> Dict[str, str]:
    """
    For each initial letter, pick the single most frequent character.
    Example: {'w': Counter({'我': 1234, '问': 200, ...})}
    -> {'w': '我'}
    """
    best: Dict[str, str] = {}
    for initial, counter in initial_to_counts.items():
        most_common_char, _ = counter.most_common(1)[0]
        best[initial] = most_common_char
    return best

def decode_initial_sequence(
    initials: str,
    initial_to_best_char: Dict[str, str],
    unknown_char: str = "□",
) -> str:
    """
    Convert an initial-letter sequence (e.g. 'wysj')
    to a Chinese string using a simple per-initial mapping.

    unknown_char: used when an initial is not in the learned mapping.
    """
    result_chars = []
    for letter in initials.lower().strip():
        result_chars.append(initial_to_best_char.get(letter, unknown_char))
    return "".join(result_chars)


# trial 
if __name__ == "__main__":
    # ===== 1. Example training data =====
    # training_texts = [
    #     "我要睡觉了。",
    #     "我喜欢中国的网络漫画。",
    #     "我想睡觉，但是我还要学习。",
    #     "要努力学习，也要好好休息。",
    # ]
    training_texts = zho_text_cleaned

    # ===== 2. Train frequency statistics =====
    initial_to_counts = build_initial_to_char_counts(training_texts)
    initial_to_best_char = build_initial_to_best_char(initial_to_counts)

    print("Most frequent char per initial:")
    for init, ch in sorted(initial_to_best_char.items()):
        print(f"{init} -> {ch}")

    # ===== 3. Decode some initial sequences =====
    examples = ["wysj", "wxsj", "yxlx"]
    for seq in examples:
        hanzi = decode_initial_sequence(seq, initial_to_best_char)
        print(f"{seq} -> {hanzi}")
