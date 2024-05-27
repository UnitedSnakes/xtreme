from datasets import load_dataset

xnli_dataset = load_dataset("xnli", "all_languages")

print(xnli_dataset)