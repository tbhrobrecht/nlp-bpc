from pathlib import Path
from pypinyin import pinyin, Style
import csv
import sys

# Increase maximum allowed CSV field size
try:
    csv.field_size_limit(sys.maxsize)
except OverflowError:
    # Fallback for platforms where sys.maxsize is too large for C long
    csv.field_size_limit(2 ** 31 - 1)  # 2147483647


def hanzi_to_pinyin(text: str, style: Style = Style.TONE, separator: str = " ") -> str:
    syllables_nested = pinyin(text, style=style, strict=False)
    syllables_flat = [item[0] for item in syllables_nested]
    return separator.join(syllables_flat)


def modify_text_column(input_path: str, output_path: str, transform_function) -> None:
    """
    Modify the `text` column of a CSV file using a transformation function.
    """
    in_path = Path(input_path)
    out_path = Path(output_path)

    with in_path.open("r", encoding="utf-8", newline="") as infile, \
         out_path.open("w", encoding="utf-8", newline="") as outfile:

        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames

        if fieldnames is None:
            raise ValueError("Input CSV has no header / fieldnames.")

        if "text" not in fieldnames:
            raise ValueError("Input CSV does not contain a 'text' column.")

        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            original_text = row["text"]
            row["text"] = transform_function(original_text)
            writer.writerow(row)


def extract_first_letters(text: str) -> str:
    """
    Extract the first letter of each word in the text and concatenate them without spaces.
    Assumes text is Pinyin already (space-separated).
    """
    return "".join([word[0] for word in text.split() if word])


if __name__ == "__main__":
    input_csv = r"C:\Users\tbhro\PycharmProjects\nlp-bpc\zho_data\first_part.csv"

    # 1) Create CSV with Hanzi → Pinyin in `text` column
    modify_text_column(
        input_csv,
        "sample_text_output.csv",
        lambda t: hanzi_to_pinyin(t, style=Style.TONE3, separator=" "),
    )

    # 2) Create CSV with only first letters of each Pinyin syllable
    modify_text_column(
        "sample_text_output.csv",
        "sample_text_first_letters.csv",
        extract_first_letters,
    )

    print("Conversion done → sample_text_output.csv")
    print("First letters file created → sample_text_first_letters.csv")
