# File: evaluate.py
# Author: Shanglin Yang
# Contact: syang662@wisc.edu

import os
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from datasets import load_dataset

from parameters import Config
from utils import Color, request_user_confirmation, ensure_directory_exists_for_file

def load_dataset_for_evaluation(task_name):
    dataset = load_dataset(task_name)
    return dataset["test"]

def evaluate_predictions(task_name):
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

    df = pd.read_csv(file_)
    print(f"Reading from {file_}\n")

    test_dataset = load_dataset_for_evaluation(task_name)
    
    # Ensure that the dataset and the CSV file have the same order and size
    assert df.shape[0] == len(test_dataset), "Row numbers mismatch"
    assert all(df["id"] == [example["id"] for example in test_dataset]), "ID columns mismatch"
    assert all(df["label"] == [example["label"] for example in test_dataset]), "Label columns mismatch"

    prompt_types = df["language"].unique()
    print("For inference results:")

    df_not_inferenced = df[df["prediction"].str.startswith("not inferenced")]
    print("Not inferenced prompts:")
    print(df_not_inferenced["language"].value_counts())
    print("\n")

    df_filtered = df[~df["prediction"].str.startswith("not inferenced")]

    for lang in prompt_types:
        df_lang = df_filtered[df_filtered["language"] == lang]
        if df_lang.empty:
            continue

        y_true = df_lang["label"]
        y_pred = df_lang["prediction"]

        report = classification_report(y_true, y_pred, output_dict=True)
        cr_file = os.path.join(Config.cr_folder, f"{lang}_classification_report.csv")
        ensure_directory_exists_for_file(cr_file)
        pd.DataFrame(report).transpose().to_csv(cr_file)
        print(f"Classification report for {lang} saved to {cr_file}")

        cm = confusion_matrix(y_true, y_pred)
        df_cm = pd.DataFrame(cm, index=np.unique(y_true), columns=np.unique(y_pred))
        cm_file = os.path.join(Config.cm_folder, f"{lang}_confusion_matrix.csv")
        ensure_directory_exists_for_file(cm_file)
        df_cm.to_csv(cm_file)
        print(f"Confusion matrix for {lang} saved to {cm_file}")

        print("----------------")
        print(f"Language: {lang}")
        print(f"Classification report:\n{classification_report(y_true, y_pred)}")
        print(f"Confusion matrix:\n{df_cm}")
        print("----------------\n")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate prediction results.")
    parser.add_argument("task_name", type=str, help="Task name for the dataset")

    args = parser.parse_args()

    evaluate_predictions(args.task_name)
