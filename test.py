import logging
logging.basicConfig(level=logging.INFO)

# from datasets import DatasetDict, load_dataset

# dataset = load_dataset("lince", "lid_spaeng", trust_remote_code=True)

# print(dataset)


ds = tfds.load('huggingface:lince/lid_spaeng')