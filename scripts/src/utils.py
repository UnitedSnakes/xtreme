# File: utils.py
# Author: Shanglin Yang
# Contact: syang662@wisc.edu

import pandas as pd
import re
import numpy as np
import os

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


def request_user_confirmation(question_prompt, yes_prompt, no_prompt, auto_confirm=None):
    if auto_confirm is not None:
        if auto_confirm.lower() == 'y':
            print(yes_prompt)
            return True
        elif auto_confirm.lower() == 'n':
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


def find_first_keyword(completion, keywords, require_immediate_label=True):
    completion_lower = completion.lower().strip()
    
    if require_immediate_label:
        if completion_lower.startswith("\""):
            completion_lower = completion_lower[1:]
            
        if completion_lower.startswith("**"):
            completion_lower = completion_lower[2:]
            
        if completion_lower.startswith(("a: ", "1. ")):
            completion_lower = completion_lower[3:]
            
        for pattern in Config.pattern:
            if completion_lower.startswith(pattern):
                return pattern
        return None
    
    keyword_positions = [
        (keyword, completion_lower.find(keyword)) for keyword in keywords
    ]
    keyword_positions = [
        (kw, pos) for kw, pos in keyword_positions if pos != -1
    ]  # Remove keywords not found in the text
    keyword_positions.sort(key=lambda x: x[1])  # Sort by position

    if keyword_positions:
        return keyword_positions[0][0]  # Return the keyword with the smallest position
    return None


def check_completion_format(df, extract=False):
    # check if completions match expected formats: in favor|against|neutral or unclear|unclear or neutral|not inferenced
    df_warnings = pd.DataFrame(columns=["id", "tweet_id", "inference_results", "level"])

    for index, (id, tweet_id, completion) in enumerate(
        zip(df["id"], df["tweet_id"], df["inference_results"])
    ):
        try:
            matches = re.findall(Config.re_pattern, completion.lower())
            if len(matches) != 1:
                print(
                    f"WARNING: Completion does not contain exactly one of {Config.pattern}.\nCompletion: {completion}"
                )
                # level = 1
                new_row = pd.DataFrame(
                    [
                        {
                            "id": id,
                            "tweet_id": tweet_id,
                            "inference_results": completion,
                            "level": 1,
                        }
                    ]
                )
                df_warnings = pd.concat([df_warnings, new_row], ignore_index=True)
                
                # automatically extract the label/pattern using the keyword that appeared first
                if extract:
                    # determine if completion starts with a keyword
                    first_keyword = find_first_keyword(completion, Config.pattern)
                    # modify df in place
                    if first_keyword:
                        df.loc[index, "inference_results"] = first_keyword

        except:
            print(
                f"WARNING: This tweet obtained erroneous inference: id={id}, tweet_id={tweet_id}, completion={completion}"
            )
            # level = 2
            new_row = pd.DataFrame(
                [
                    {
                        "id": id,
                        "tweet_id": tweet_id,
                        "inference_results": completion,
                        "level": 2,
                    }
                ]
            )
            df_warnings = pd.concat([df_warnings, new_row], ignore_index=True)

    return df_warnings


def encode_stance_as_int(series):
    # check if stances match expected formats: in favor|against|neutral or unclear|unclear or neutral|not inferenced
    def map_stance_to_int(stance):
        if Config.pattern[0] in stance:
            return 1  # in favor
        elif Config.pattern[1] in stance:
            return 2  # against
        elif Config.pattern[2] in stance or Config.pattern[3] in stance:
            return 3  # neutral
        elif Config.pattern[4] in stance:
            # raise ValueError(f"Ill format: stance={stance}")
            return 0  # not inferenced
        else:
            raise ValueError(f"Ill format: stance={stance}")

    # for i, stance in enumerate(series["inference_results"]):
    #     inference_matrix[i // Config.test_size, i % Config.test_size] = int_

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
