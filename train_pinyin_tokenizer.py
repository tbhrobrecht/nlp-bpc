#!/usr/bin/env python
"""
Train a SentencePiece BPE tokenizer for a Mandarin pinyin-initials corpus.

The corpus is assumed to be in a CSV-like format where each line looks like:

"pinyin initials and punctuation ...",cleaned_CTS-...,child-available-speech,...,zho

This script:
1. Reads the file line by line.
2. Extracts ONLY the first field (the big quoted pinyin-initials text).
3. Writes those texts to a temporary plain-text file.
4. Trains a SentencePiece BPE tokenizer on that text.

Output:
    <model_prefix>.model
    <model_prefix>.vocab
"""

import os
import csv
import tempfile
import sentencepiece as spm


# ========= CONFIGURE THESE =========
INPUT_CORPUS = "sample_text_first_letters.csv"      # your file containing lines like the example you pasted
MODEL_PREFIX = "zho_model_data/pinyin_bpe"          # will produce pinyin_bpe.model / pinyin_bpe.vocab
VOCAB_SIZE = 500                     # adjust as needed (500–2000 is usually plenty)
# ===================================


def extract_text_column_to_temp(input_path: str) -> str:
    """
    Extract the first column (pinyin-initial text) from a CSV-like file
    into a temporary plain-text file (one utterance per line).

    Returns the path to the temporary file.
    """
    # Create a temp file that SentencePiece will read from
    temp_fd, temp_path = tempfile.mkstemp(prefix="pinyin_corpus_", suffix=".txt")
    os.close(temp_fd)  # We'll reopen it normally

    with open(input_path, "r", encoding="utf-8") as f_in, open(
        temp_path, "w", encoding="utf-8"
    ) as f_out:
        # Use csv.reader to correctly handle quotes and commas
        reader = csv.reader(f_in)

        for row in reader:
            if not row:
                continue
            text = row[0]  # first column = pinyin-initials + punctuation

            # Optional: strip surrounding whitespace
            text = text.strip()

            # If your file has surrounding quotes kept by csv, it's already handled.
            # Just write one utterance per line:
            if text:
                f_out.write(text + "\n")

    return temp_path


def train_bpe_tokenizer(corpus_txt_path: str, model_prefix: str, vocab_size: int):
    """
    Train SentencePiece BPE tokenizer on the prepared plain-text corpus.
    """
    out_dir = os.path.dirname(model_prefix)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print("Training BPE tokenizer...")
    print(f"  Corpus file : {corpus_txt_path}")
    print(f"  Model prefix: {model_prefix}")
    print(f"  Vocab size  : {vocab_size}")
    print()

    spm.SentencePieceTrainer.train(
        input=corpus_txt_path,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type="bpe",       # 🔴 Byte Pair Encoding
        character_coverage=1.0, # includes ASCII + '。' etc. safely

        # Special token IDs
        unk_id=0,
        bos_id=1,
        eos_id=2,
        pad_id=3,

        # No extra special symbols for now
        user_defined_symbols=[],

        # For small alphabets (a–z + punctuation), allow slight variation
        hard_vocab_limit=False,
    )

    print("✅ Tokenizer training complete!")
    print(f"  Saved: {model_prefix}.model")
    print(f"         {model_prefix}.vocab")


def main():
    # 1. Extract first column (pinyin initials) into a clean text file
    corpus_txt = extract_text_column_to_temp(INPUT_CORPUS)

    # 2. Train BPE tokenizer on that text
    train_bpe_tokenizer(corpus_txt, MODEL_PREFIX, VOCAB_SIZE)

    # 3. Optionally keep the temp corpus text
    # If you want to inspect it, comment out the unlink:
    # print("Temp corpus file at:", corpus_txt)
    os.unlink(corpus_txt)


if __name__ == "__main__":
    main()
