# File: utils.py
# Author: Shanglin Yang
# Contact: syang662@wisc.edu

from datetime import datetime
import logging
import pandas as pd
import re
import numpy as np
import os
from typing import Any
from datasets import DatasetDict, load_dataset

from parameters import Config


class Color:
    PURPLE = "\033[95m"
    CYAN = "\033[96m"
    DARKCYAN = "\033[36m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    END = "\033[0m"


def request_user_confirmation(
    question_prompt, yes_prompt, no_prompt, auto_confirm=None
):
    if auto_confirm is not None:
        if auto_confirm.lower() == "y":
            print(yes_prompt)
            return True
        elif auto_confirm.lower() == "n":
            print(no_prompt)
            return False

    while True:
        user_input = input(question_prompt)
        if user_input.lower() == "y":
            print(yes_prompt)
            return True
        elif user_input.lower() == "n":
            print(no_prompt)
            return False
        else:
            print("Invalid input. Please try again.")


def find_first_keyword(prediction, keywords, require_immediate_label=True):
    # if prediction.startswith("1."):
    #     print("here")
    prediction_lower = prediction.lower().strip()

    if require_immediate_label:
        if prediction_lower.startswith(('"', "(")):
            prediction_lower = prediction_lower[1:]

        if prediction_lower.startswith("**"):
            prediction_lower = prediction_lower[2:]

        if prediction_lower.startswith(("a: ")):
            prediction_lower = prediction_lower[3:]

        for pattern in Config.patterns:
            if prediction_lower.startswith(pattern):
                return pattern[0]
        return None

    keyword_positions = [
        (keyword, prediction_lower.find(keyword)) for keyword in keywords
    ]
    keyword_positions = [
        (kw, pos) for kw, pos in keyword_positions if pos != -1
    ]  # Remove keywords not found in the text
    keyword_positions.sort(key=lambda x: x[1])  # Sort by position

    if keyword_positions:
        return keyword_positions[0][0]  # Return the keyword with the smallest position
    return None


def check_prediction_format(df, extract=False, verbose=True):
    # check if predicitons match expected formats: 0|1|2|not inferenced
    df_warnings = pd.DataFrame(columns=["index", "prediction", "level"])

    if Config.dataset_name == "tatoeba":
        return df_warnings

    for index, (id, prediction) in enumerate(zip(df["index"], df["prediction"])):
        try:
            matches = [
                re.findall(r"\b(" + "|".join(pattern) + r")\b", prediction.lower())
                for pattern in Config.patterns
            ]
            if sum(len(match) for match in matches) != 1:
                print(
                    f"WARNING: Prediction does not contain exactly one of {Config.patterns}.\nPrediction: {prediction}"
                )
                # level = 1
                new_row = pd.DataFrame(
                    [
                        {
                            "index": id,
                            "prediction": prediction,
                            "level": 1,
                        }
                    ]
                )
                df_warnings = pd.concat([df_warnings, new_row], ignore_index=True)

            # automatically extract the label/pattern using the keyword that appeared first
            if extract:
                # determine if prediction starts with a keyword
                # if df.loc[index, "prediction"] == "(1).":
                #     print("hey")
                first_keyword = find_first_keyword(prediction, Config.patterns)
                # modify df in place
                if first_keyword:
                    df.loc[index, "prediction"] = first_keyword
                    # if type(df.loc[index, "prediction"]) != str:
                    # if first_keyword == "1.":
                    # print("hey")
                else:
                    df.loc[index, "prediction"] = "unparsable"

        except:
            if verbose:
                print(
                    f"WARNING: This tweet obtained erroneous inference: id={id}, prediction={prediction}"
                )
            # level = 2
            new_row = pd.DataFrame(
                [
                    {
                        "index": id,
                        "prediction": prediction,
                        "level": 2,
                    }
                ]
            )
            df_warnings = pd.concat([df_warnings, new_row], ignore_index=True)

            if extract:
                df.loc[index, "prediction"] = "unparsable"

    return df_warnings


def encode_stance_as_int(series):
    # check if stances match expected formats: 0|1|2|not inferenced
    def map_stance_to_int(stance):
        for p in Config.patterns[0]:
            if p in stance:
                return 0  # entailment
        for p in Config.patterns[1]:
            if p in stance:
                return 1  # neutral
        for p in Config.patterns[2]:
            if p in stance:
                return 2  # contradiction
        for p in Config.patterns[3]:
            if p in stance:
                return 3  # not inferenced
        raise ValueError(f"Ill format: stance={stance}")

    mapped_results = series.str.lower().apply(map_stance_to_int)

    matrix = np.array(mapped_results).reshape((-1, Config.test_size))
    print("Encoded matrix shape:")
    print(matrix.shape)
    print("\n")
    assert matrix.shape == (len(series) / Config.test_size, Config.test_size)

    return matrix


def ensure_directory_exists_for_file(filepath):
    """
    Ensure the parent directory for a given file path exists.

    This function checks if the parent directory of the specified file path exists. If the directory does not exist, it creates the directory (including any necessary intermediate directories) to ensure that the file path is valid for file operations such as file creation or writing.

    Parameters:
    - filepath (str): The complete file path for which the parent directory should be verified or created.
    """
    directory = os.path.dirname(filepath)
    if not os.path.exists(directory):
        os.makedirs(directory)


def split_dict_in_half(original_dict):
    keys = list(original_dict.keys())
    half_size = len(keys) // 2

    first_half_dict = {
        key: original_dict[key][: len(original_dict[key]) // 2] for key in keys
    }
    second_half_dict = {
        key: original_dict[key][len(original_dict[key]) // 2 :] for key in keys
    }

    return first_half_dict, second_half_dict


def add_index_column(dataset: DatasetDict) -> DatasetDict:
    def add_index(example, idx):
        if "id" not in example:
            example["id"] = idx
        return example

    return dataset.map(
        add_index,
        with_indices=True,
        batched=True,
        num_proc=Config.preprocess_threads,
    )

def swap_sentences(example):
    example["source_sentence"], example["target_sentence"] = example["target_sentence"], example["source_sentence"]
    return example


def load_dataset_by_name(
    language: str,
    split: str = "test",
    switch_source_and_target: bool = False,
) -> DatasetDict:
    if Config.dataset_name == "xnli":
        dataset = load_dataset(Config.dataset_name, language, split=split)
    elif Config.dataset_name == "tatoeba":
        dataset = load_dataset("xtreme", f"tatoeba.{Config.lang2_dict[language]}")[
            "validation"
        ]

        if len(dataset) != 1000:
            raise ValueError("Tatoeba dataset should have 1000 examples.")

        # switch source and target if necessary
        if switch_source_and_target:
            dataset = dataset.map(swap_sentences)
    elif Config.dataset_name == "multi_nli":
        dataset = load_dataset(Config.dataset_name, split=split)
    else:
        raise NotImplementedError(f"Dataset {Config.dataset_name} not supported.")

    if Config.is_test_run:
        dataset = dataset.select(range(Config.test_size))

    return dataset

def get_latest_checkpoint_dir(base_dir):
    sub_dirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    sub_dirs = [d for d in sub_dirs if d.split(os.sep)[-1].replace("_", "").isdigit()]
    
    # Add handling for the full timestamp
    latest_sub_dir = max(sub_dirs, key=lambda d: datetime.strptime(d.split(os.sep)[-1], "%Y%m%d_%H%M%S"))
    return latest_sub_dir

class LoggingFormatter(logging.Formatter):
    def format(self, record):
        msg = super().format(record)
        return msg.replace('|', '\n')

class ColorFormatter(logging.Formatter):
    def format(self, record):
        if record.levelno == logging.INFO:
            record.msg = f"\033[92m{record.msg}\033[0m"  # Green
        elif record.levelno == logging.WARNING:
            record.msg = f"\033[93m{record.msg}\033[0m"  # Yellow
        elif record.levelno == logging.ERROR:
            record.msg = f"\033[91m{record.msg}\033[0m"  # Red
        return super().format(record)
    
class MultilineFormatter(LoggingFormatter):
    def format(self, record):
        message = super().format(record)
        return '\n'.join([line.strip() for line in message.splitlines()])
    
# Redirect print to logging while keeping terminal output
class PrintLogger:
    def __init__(self, terminal, log_func):
        self.terminal = terminal
        self.log_func = log_func

    def write(self, message):
        self.terminal.write(message)
        self.log_func(message.strip())

    def flush(self):
        self.terminal.flush()