# File: evaluate.py
# Author: Shanglin Yang
# Contact: syang662@wisc.edu

import os
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
# import logging

from parameters import *
from utils import *

# logger = logging.getLogger('evaluation')
# logger.setLevel(logging.INFO)

# file_handler = logging.FileHandler('evaluation.log')
# file_handler.setLevel(logging.INFO)
# file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
# file_handler.setFormatter(file_formatter)
# logger.addHandler(file_handler)

# stream_handler = logging.StreamHandler()
# stream_handler.setLevel(logging.INFO)
# stream_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
# stream_handler.setFormatter(stream_formatter)
# logger.addHandler(stream_handler)

# logger.info("This message will go to the log file and the console.")



if not os.path.isfile(Config.annotated_results_file):
    confirmation_message = f"Annotated file {Config.annotated_results_file} does not exist. Are you sure to continue on raw results? Ill-formatted problems may occur. {Color.YELLOW}Continue?{Color.END} [Y/n]: "

    confirmed = request_user_confirmation(
        confirmation_message,
        f"Aye captain. Continue on raw file {Config.results_file}. Will discard ill-formatted completions if any.",
        "Terminated.",
    )

    if not confirmed:
        exit(0)

    file_ = Config.results_file
else:
    file_ = Config.annotated_results_file

# try:
#     df = pd.read_csv(file_)
#     print(f"Reading from {file_}\n")
# except pd.errors.ParserError as e:
#     print("ParserError:", e)
#     with open(file_, 'r') as file:
#         for i, line in enumerate(file):
#             if i == 6818:
#                 print("Error in line:", i)
#                 print(line)
#                 break

df = pd.read_csv(file_)
print(f"Reading from {file_}\n")

# all completions should adhere to desired formats after being annotated
assert check_completion_format(df).shape[0] == 0

# assert row numbers match in dataset, raw results and annotated results
df_dataset = pd.read_csv(Config.complete_dataset_file)
df_raw_results = pd.read_csv(Config.results_file)
assert df.shape[0] == df_dataset.shape[0] == df_raw_results.shape[0]
assert df["tweet_id"].equals(df_dataset["tweet_id"]) and df["tweet_id"].equals(
    df_raw_results["tweet_id"]
)
assert df["true_label"].equals(df_dataset["true_label"]) and df["true_label"].equals(
    df_raw_results["true_label"]
)

# assert prompt versions correct

assert (
    df["prompt_type"].iloc[: Config.test_size]
    == df_dataset["prompt_type"].iloc[: Config.test_size]
).all(), "df and df_dataset prompt_type mismatch"

assert (
    df_dataset["prompt_type"].iloc[: Config.test_size]
    == df_raw_results["prompt_type"].iloc[: Config.test_size]
).all(), "df_dataset and df_raw_results prompt_type mismatch"

assert (
    tuple(df_raw_results["prompt_type"].iloc[: 2 * Config.test_size])
    == Config.prompt_version_list + Config.prompt_version_list
), "df_raw_results and Config prompt_version_list mismatch"

# expect df to be (9688, 7)

prompt_types = Config.prompt_version_list
print("For inference results:")
inference_matrix = encode_stance_as_int(df["inference_results"])  # (674, 26)

# notice that both "unclear or neutral" and "neutral or unclear" appeared in inference results. Both of them also appeared in prompts. They are counted as the same class here.
# print(f"\n{inference_matrix}\n")
unique_elements, counts = np.unique(inference_matrix, return_counts=True)
print("Unfiltered inference results (contain 'not inferenced'):")
for element, count in zip(unique_elements, counts):
    print(f"{Config.pattern[element - 1]}: {count} times")
print("\n")

# stats about rows that were "not inferenced"
df_not_inferenced = df[df["inference_results"].str.startswith("not inferenced")]
print("Not inferenced prompts:")
print(df_not_inferenced["prompt_type"].value_counts())
print("\n")

# filter out rows that were not inferenced
# print(df_not_inferenced.head(26))
# print(df_not_inferenced.shape)
# print(df.head(26))
# print(df.shape)
# df -= df_not_inferenced

# df_merged = pd.merge(
#     df,
#     df_not_inferenced,
#     on=["tweet_id", "prompt_type"],
#     how="left",
#     indicator=True,
#     suffixes=("", "_drop"),
# )
# df_merged = df_merged[df_merged["_merge"] == "left_only"].drop(columns="_merge")
# df = df_merged.loc[:, ~df_merged.columns.str.endswith('_drop')]
# print(df.head(26))
# print(df.shape)

# inference_matrix = encode_stance_as_int(df["inference_results"])  # (674, 26)

# unique_elements, counts = np.unique(inference_matrix, return_counts=True)
# print("Filtered inference results:")
# for element, count in zip(unique_elements, counts):
#     print(f"{Config.pattern[element - 1]}: {count} times")

# input()

print("For true labels:")
true_matrix = encode_stance_as_int(df["true_label"])  # (692, 14)

unique_elements, counts = np.unique(true_matrix, return_counts=True)
print("True labels:")
for element, count in zip(unique_elements, counts):
    print(f"{Config.pattern[element - 1]}: {count} times")
print("\n")

for i in range(Config.test_size):
    inference_ints = inference_matrix[:, i]
    actual_ints = true_matrix[:, i]

    mask = inference_ints != 0
    
    filtered_actual_ints = actual_ints[mask]
    filtered_inference_ints = inference_ints[mask]
    
    print(
        f"Filtered {np.count_nonzero(mask == 0)} rows that were not infereced under {Config.prompt_version_list[i]}."
    )

    df_report = pd.DataFrame(
        classification_report(
            filtered_actual_ints,
            filtered_inference_ints,
            labels=[1, 2, 3],
            target_names=Config.pattern[:3],
            output_dict=True,
        )
    )

    cr_file = os.path.join(Config.cr_folder, prompt_types[i] + ".csv")
    ensure_directory_exists_for_file(cr_file)
    df_report.to_csv(cr_file)

    cm = confusion_matrix(
        filtered_actual_ints, filtered_inference_ints, labels=[1, 2, 3]
    )
    df_cm = pd.DataFrame(
        cm,
        index=[f"True {label}" for label in Config.pattern[:3]],
        columns=[f"Inferenced {label}" for label in Config.pattern[:3]],
    )

    cm_file = os.path.join(Config.cm_folder, prompt_types[i] + ".csv")
    ensure_directory_exists_for_file(cm_file)
    df_cm.to_csv(os.path.join(Config.cm_folder, prompt_types[i] + ".csv"))

    print("----------------")
    print(f"id: {i + 1}")
    print(f"prompt type: {prompt_types[i]}")

    print(f"F1 macro: {df_report['macro avg']['f1-score']}\n")
    print(f"Confusion matrix:")
    print(df_cm)
    print(f"Class labels: [0, 1, 2] = {Config.pattern[:3]}")
    print(f"rows: true labels")
    print(f"columns: predicted labels\n")

    print("Classification report:")
    print(df_report)

    print("----------------\n")
