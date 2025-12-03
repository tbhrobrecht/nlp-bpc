from typing import Literal
from pypinyin import pinyin, Style


PinyinStyle = Literal["tone_marks", "tone_numbers", "no_tone"]


def hanzi_to_pinyin(text: str, style: PinyinStyle = "tone_marks", separator: str = " ") -> str:
    """
    Convert a Chinese text (Hanzi) to Pinyin.

    Parameters
    ----------
    text : str
        Input Chinese text.
    style : {"tone_marks", "tone_numbers", "no_tone"}
        Pinyin style:
        - "tone_marks": nǐ hǎo
        - "tone_numbers": ni3 hao3
        - "no_tone": ni hao
    separator : str
        String used to join syllables.

    Returns
    -------
    str
        Converted Pinyin string.
    """
    if style == "tone_marks":
        pinyin_style = Style.TONE  # with tone marks
    elif style == "tone_numbers":
        pinyin_style = Style.TONE3  # with tone numbers
    elif style == "no_tone":
        pinyin_style = Style.NORMAL  # no tones
    else:
        raise ValueError(f"Unknown style: {style}")

    # pinyin(...) returns a list of lists, e.g. [['ni3'], ['hao3']]
    syllables_nested = pinyin(text, style=pinyin_style, strict=False)
    syllables_flat = [item[0] for item in syllables_nested]

    return separator.join(syllables_flat)


if __name__ == "__main__":
    sentence = "我喜欢中国的网络漫画。"
    print("Original:", sentence)
    print("With tone marks:", hanzi_to_pinyin(sentence, style="tone_marks"))
    print("With tone numbers:", hanzi_to_pinyin(sentence, style="tone_numbers"))
    print("No tone:", hanzi_to_pinyin(sentence, style="no_tone"))

