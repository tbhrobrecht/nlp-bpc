https://huggingface.co/learn/llm-course/en/chapter6/5 to read about byte-pair encoding tokenization 

file_to_pinyin => to convert files to pinyin, i/o: sample_text/sample_text_output

character_mapping => combines pinyin_frequency with character_frequency_heatmap
hanzi_pinyin => to convert characters to corresponding pinyin 
pinyin_frequency => frequency mapping of initial letter to corresponding character
character_frequency_heatmap => connected with the above to visualise cross referencing heatmap 

resolving duplicates between single- and multi-string characters e.g., 一、个 and 一个
P(char | initial = "y") is built from only those y-positions that aren’t “claimed” by a known pattern, so it reflects “true single-syllable y choices” like 有 要 也 etc.
P(word | pattern = "yg") is a separate distribution over words like 一个, 一共, …