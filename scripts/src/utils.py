# File: utils.py
# Author: Shanglin Yang
# Contact: syang662@wisc.edu

import random
import pandas as pd
import re
import numpy as np
import os
from typing import Any
from datasets import DatasetDict, Dataset, load_dataset, Features, Value

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
        # if index == 57:
        #     print("hereeee")
        # if prediction == "N":
        #     print("hey")
        try:
            # if type(prediction) != float: # not nan
            #     # print("hey")
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

    first_half_keys = keys[:half_size]
    second_half_keys = keys[half_size:]

    first_half_dict = {
        key: original_dict[key][: len(original_dict[key]) // 2] for key in keys
    }
    second_half_dict = {
        key: original_dict[key][len(original_dict[key]) // 2 :] for key in keys
    }

    return first_half_dict, second_half_dict


def add_index_column(dataset: DatasetDict) -> DatasetDict:
    def add_index(example, idx):
        if "index" not in example:
            example["index"] = idx
        return example

    return dataset.map(
        add_index,
        with_indices=True,
        batched=True,
        num_proc=Config.preprocess_threads,
    )


def rename_column(example, lang2, lang3):
    new_example = {
        key if key == lang2 else lang3: value for key, value in example.items()
    }
    assert lang3 in example
    example[lang2] = example.pop(lang3)
    example = new_example
    # print(example.keys())
    return example


def restructure_dataset(example):
    language = list(example["translation"][0].keys())[1]
    new_example = {
        "index": example["index"],
        "en": [ex["en"] for ex in example["translation"]],
        language: [ex[language] for ex in example["translation"]],
    }
    return new_example


def load_dataset_by_name(
    language: str, split: str = "test", add_index: bool = True
) -> DatasetDict:
    if Config.dataset_name == "xnli":
        dataset = load_dataset(Config.dataset_name, language, split=split)
    elif Config.dataset_name == "tatoeba":
        combinations = [
            {"lang1": "en", "lang2": language},
            {"lang1": language, "lang2": "en"},
        ]
        if language in Config.lang2_dict:
            combinations.append({"lang1": Config.lang2_dict[language], "lang2": "en"})
            combinations.append({"lang1": "en", "lang2": Config.lang2_dict[language]})

        dataset = None
        for combo in combinations:
            try:
                dataset = load_dataset(
                    "tatoeba",
                    lang1=combo["lang1"],
                    lang2=combo["lang2"],
                    trust_remote_code=True,
                )["train"]

                # pick 1000 random examples for each language
                assert len(dataset) >= 1000, f"Not enough examples in dataset. Expected at least 1000. Obtained: {len(dataset)}"
                
                with open(Config.random_seed_path, "r") as f:
                    seed = int(f.read().strip())

                random.seed(seed)

                sampled_dataset = random.sample(list(dataset), 1000)
                
                for ex in sampled_dataset:
                    ex["index"] = int(ex.pop("id"))
                    
                # # fo
                # # print(sampled_dataset)
                    
                # for ex 
                # # sampled_dataset = [ex for ex in sampled_dataset]
                    
                dataset = Dataset.from_dict(
                    {key: [d[key] for d in sampled_dataset] for key in sampled_dataset[0]}
                )

                break
            except Exception as e:
                print(e)
                continue

        if dataset is None:
            raise ValueError(
                f"Dataset {Config.dataset_name} not found for language {language}."
            )

        # make sure en is first in the translation dict
        features = Features({
            'index': Value('int64'),
            'translation': {
                'en': Value('string'),
                combo["lang1"]: Value('string')
            }
        })
                    
        if combo["lang1"] != "en":
            data_dict = dataset.to_dict()
            reordered_data_dict = {
                "index": data_dict["index"],
                "translation": [
                    {"en": ex["en"], combo["lang1"]: ex[combo["lang1"]]}
                    for ex in data_dict["translation"]
                ],
            }

            dataset = Dataset.from_dict(reordered_data_dict, features=features)

        # defactor translation column
        dataset = dataset.map(
            restructure_dataset,
            remove_columns=["translation"],
            batched=True,
            num_proc=Config.preprocess_threads,
        )

        # rename the second language using 2 letters if necessary
        keys = list(dataset[0].keys())
        if language in Config.lang2_dict and len(keys[2]) == 3:
            dataset = dataset.map(
                rename_column,
                fn_kwargs={"lang2": language, "lang3": Config.lang2_dict[language]},
                batched=True,
                num_proc=Config.preprocess_threads,
            )

    else:
        raise NotImplementedError(f"Dataset {Config.dataset_name} not supported.")

    if Config.is_test_run:
        dataset = dataset.select(range(Config.test_size))

    if add_index:
        dataset = add_index_column(dataset)

    return dataset
