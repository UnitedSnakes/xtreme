import os
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from datasets import load_dataset

from parameters import Config
from utils import Color, check_prediction_format, load_dataset_by_name, request_user_confirmation, ensure_directory_exists_for_file

if Config.dataset_name == "xnli":
    label_mapping = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}
elif Config.dataset_name == "tatoeba":
    pass
else:
    raise NotImplementedError(f"Dataset {Config.dataset_name} not supported.")

def evaluate_predictions(language):
    if Config.dataset_name == "xnli" and not os.path.isfile(Config.annotated_results_file.replace(".csv", f"_{language}.csv")):
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
        if Config.dataset_name == "xnli":
            file_ = Config.annotated_results_file.replace(".csv", f"_{language}.csv")
        elif Config.dataset_name == "tatoeba":
            file_ = Config.results_file.replace(".csv", f"_{language}.csv")
        else:
            raise NotImplementedError(f"Dataset {Config.dataset_name} not supported.")

    try:
        df = pd.read_csv(file_, dtype={'prediction': 'str'})
    except Exception as e:
        print(e)
        return None
    print(f"Reading from {file_}\n")

    test_dataset = load_dataset_by_name(language, switch_source_and_target=Config.swap_sentences)
    
    assert df.shape[0] == len(test_dataset), f"Row numbers mismatch: {df.shape[0]} vs {len(test_dataset)}"
    
    if (df["language"] != language).any():
        raise ValueError("Language column in the CSV file does not match the specified language.")
    
    if Config.dataset_name == "xnli":
        assert all(df["label"] == [example["label"] for example in test_dataset]), "Label columns mismatch"
        
        print("For inference results:")
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
        
        df_warnings = check_prediction_format(df, verbose=False)
        df = df[~df['index'].isin(df_warnings['index'])]
    
        y_true = df_filtered["label"]
        y_pred = df_filtered["prediction"]

        y_true = y_true.map(label_mapping)
        y_pred = y_pred.map(label_mapping)

        report = classification_report(y_true, y_pred, output_dict=True)
        cr_file = os.path.join(Config.cr_dir, f"{language}_classification_report.csv")
        ensure_directory_exists_for_file(cr_file)
        pd.DataFrame(report).transpose().to_csv(cr_file)
        print(f"Classification report for {language} saved to {cr_file}")

        cm = confusion_matrix(y_true, y_pred, labels=list(label_mapping.values()))
        df_cm = pd.DataFrame(cm, index=list(label_mapping.values()), columns=list(label_mapping.values()))
        cm_file = os.path.join(Config.cm_dir, f"{language}_confusion_matrix.csv")
        ensure_directory_exists_for_file(cm_file)
        df_cm.to_csv(cm_file)
        print(f"Confusion matrix for {language} saved to {cm_file}")

        print("----------------")
        print(f"Language: {language}")
        print(f"Classification report:\n{classification_report(y_true, y_pred)}")
        print(f"Confusion matrix:\n{df_cm}")
        print("----------------\n")
        
        return None
        
    elif Config.dataset_name == "tatoeba":
        # print(df["id"])
        # print([example["id"] for example in test_dataset])
        # input()
        assert all(df["id"] == [example["id"] for example in test_dataset]), "id columns mismatch"
        
        df["prediction"] = df["prediction"].astype("int64")
        
        df_oversized = df[df["prediction"] == -1]
        print("Oversized results:")
        print(df_oversized["language"].value_counts())
        print("\n")
        
        df_filtered = df.drop(df_oversized.index)
        
        accuracy = sum(df_filtered["prediction"]) / len(df_filtered["prediction"])
        support = len(df_filtered["prediction"])
        df_accuracy = pd.DataFrame({"accuracy": [accuracy], "support": [support]}, index=[0])
        accuracy_file = os.path.join(Config.metrics_dir, language + "_accuracy.csv")
        ensure_directory_exists_for_file(accuracy_file)
        df_accuracy.to_csv(accuracy_file)
        print("----------------")
        print(f"Language: {language}")
        print(f"Accuracy:\n{accuracy}")
        print("----------------\n")
        
        return accuracy
        
        
def main():
    results = []
        
    for language in Config.evaluate_languages:
        result = evaluate_predictions(language)
        if result:
            results.append(result)
        
    if Config.dataset_name == "tatoeba":
        accuracy_avg = sum(results) / len(results)
        support = len(results)
        df_accuracy_avg = pd.DataFrame({"accuracy_avg": [accuracy_avg], "support": [support]}, index=[0])
        df_accuracy_avg_file = os.path.join(Config.metrics_dir, "accuracy_avg.csv")
        ensure_directory_exists_for_file(df_accuracy_avg_file)
        df_accuracy_avg.to_csv(df_accuracy_avg_file)
        
        print("----------------")
        print(f"Accuracy avg:\n{accuracy_avg}")
        print(f"Support:\n{support}")
        print("----------------\n")
        

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate prediction results.")
    # parser.add_argument("Config.dataset_name", type=str, help="Task name for the dataset")

    args = parser.parse_args()
    
    main()
