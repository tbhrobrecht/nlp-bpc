from pathlib import Path
from pypinyin import pinyin, Style


def hanzi_to_pinyin(text: str, style: Style = Style.TONE, separator: str = " ") -> str:
    syllables_nested = pinyin(text, style=style, strict=False)
    syllables_flat = [item[0] for item in syllables_nested]
    return separator.join(syllables_flat)


def convert_file_to_pinyin(
    input_path: str,
    output_path: str,
    style: Style = Style.TONE,
    separator: str = " ",
) -> None:
    in_path = Path(input_path)
    out_path = Path(output_path)

    text = in_path.read_text(encoding="utf-8")
    converted = hanzi_to_pinyin(text, style=style, separator=separator)
    out_path.write_text(converted, encoding="utf-8")


if __name__ == "__main__":
    convert_file_to_pinyin(
        "C:\\Users\\tbhro\\PycharmProjects\\nlp-bpc\\zho_data\\sample_text.csv",
        "sample_text_output.csv",
        style=Style.TONE3,  # or Style.TONE / Style.NORMAL
        separator=" ",
    )
    print("Conversion done → sample_text_output.csv")
