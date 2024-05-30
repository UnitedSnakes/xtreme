# File: evaluate.py
# Author: Shanglin Yang
# Contact: syang662@wisc.edu

import os
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from datasets import load_dataset

from parameters import Config
from utils import Color, check_prediction_format, request_user_confirmation, ensure_directory_exists_for_file

def load_dataset_for_evaluation(task_name, language):
    dataset = load_dataset(task_name, language)
    if Config.is_test_run:
        dataset["test"] = dataset["test"].select(range(Config.test_size))
    return dataset["test"]

def evaluate_predictions(task_name, language):
    if not os.path.isfile(Config.annotated_results_file.replace(".csv", f"_{language}.csv")):
        confirmation_message = f"Annotated file {Config.annotated_results_file.replace(".csv", f"_{language}.csv")} does not exist. Are you sure to continue on raw results? Ill-formatted problems may occur. {Color.YELLOW}Continue?{Color.END} [Y/n]: "

        confirmed = request_user_confirmation(
            confirmation_message,
            f"Aye captain. Continue on raw file {Config.results_file}. Will discard ill-formatted predictions if any.",
            "Terminated.",
        )

        if not confirmed:
            exit(0)

        file_ = Config.results_file.replace(".csv", f"_{language}.csv")
    else:
        file_ = Config.annotated_results_file.replace(".csv", f"_{language}.csv")

    df = pd.read_csv(file_, dtype={'prediction': 'str'})
    print(f"Reading from {file_}\n")

    # for i in df["prediction"]:
    #     if type(i) != str:
    #         print(i)
    #         print(type(i))
    # print(language)
    # input()
    test_dataset = load_dataset_for_evaluation(task_name, language)
    
    # Ensure that the dataset and the CSV file have the same order and size
    assert df.shape[0] == len(test_dataset), f"Row numbers mismatch: {df.shape[0]} vs {len(test_dataset)}"
    # assert all(df["id"] == [example["id"] for example in test_dataset]), "ID columns mismatch"
    assert all(df["label"] == [example["label"] for example in test_dataset]), "Label columns mismatch"
    
    # all predictions should adhere to desired formats after being annotated
    # assert check_prediction_format(df).shape[0] == 0
    df_warnings = check_prediction_format(df, verbose=False)
    df = df[~df['id'].isin(df_warnings['id'])]
        
    print("For inference results:")
    # print(df["prediction"])
    # input()
    # for i in df["prediction"]:
    #     if type(i) != str:
    #         print(i)
    #         print(type(i))
    #         input()

    df_not_inferenced = df[df["prediction"].str.startswith("not inferenced")]
    print("Not inferenced prompts:")
    print(df_not_inferenced["language"].value_counts())
    print("\n")

    df_unparsable = df[df["prediction"].str.startswith("unparsable")]
    print("Unparsable results:")
    print(df_unparsable["language"].value_counts())
    print("\n")

    df_filtered = df[~df["prediction"].str.startswith(("not inferenced", "unparsable"))]

    df_filtered["prediction"] = df_filtered["prediction"].astype("int64")
    
    if (df_filtered["language"] != language).any():
        raise ValueError("Language column in the CSV file does not match the specified language.")

    y_true = df_filtered["label"]
    y_pred = df_filtered["prediction"]
    
    print(y_true)
    print(type(y_true.values))
    print(y_pred)
    print(type(y_pred.values))

    report = classification_report(y_true, y_pred, output_dict=True)
    cr_file = os.path.join(Config.cr_dir, f"{language}_classification_report.csv")
    ensure_directory_exists_for_file(cr_file)
    pd.DataFrame(report).transpose().to_csv(cr_file)
    print(f"Classification report for {language} saved to {cr_file}")

    cm = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cm, index=np.unique(y_true), columns=np.unique(y_pred))
    cm_file = os.path.join(Config.cm_dir, f"{language}_confusion_matrix.csv")
    ensure_directory_exists_for_file(cm_file)
    df_cm.to_csv(cm_file)
    print(f"Confusion matrix for {language} saved to {cm_file}")

    print("----------------")
    print(f"Language: {language}")
    print(f"Classification report:\n{classification_report(y_true, y_pred)}")
    print(f"Confusion matrix:\n{df_cm}")
    print("----------------\n")
        
def main():
    for language in Config.evaluate_languages:
        evaluate_predictions(Config.dataset_name, language)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate prediction results.")
    # parser.add_argument("task_name", type=str, help="Task name for the dataset")

    args = parser.parse_args()
    
    main()

