# File: parameters.py
# Author: Shanglin Yang
# Contact: syang662@wisc.edu

import os
import torch as th

# --------------------modify below------------------------
# --------------------------------------------------------
IS_TEST_RUN = False
IS_FINE_TUNE = False
SAVE_LOGITS = False

MODEL_ABBR = "FLAN_UL2"
MODEL = "google/flan-ul2"

# MODEL_ABBR = "MBERT"
# MODEL = "bert-base-multilingual-cased"

BATCH_SIZE = 8
MAX_OUTPUT_LENGTH = 200

# flan-ul2 = 40GB, so run 2 instances
INFERENCE_THREADS = 2
TRAIN_THREADS = 1
PREPROCESS_THREADS = 64

# https://huggingface.co/datasets/xnli
DATASET_NAME = "xnli"

# --------------------modify above------------------------
# --------------------------------------------------------

if DATASET_NAME == "xnli":
    FINETUNE_LANGUAGE = "en"
    EVALUATE_LANGUAGES = [
        "en",
        "fr",
        "es",
        "de",
        "el",
        "bg",
        "ru",
        "tr",
        "ar",
        "vi",
        "th",
        "zh",
        "hi",
        "sw",
        "ur",
    ]

DATE = ""


COMPLETE_RESULTS_FILENAME = f"all_results_{DATE}.csv"
TEST_RESULTS_FILENAME = f"test_results_{DATE}.csv"
COMPLETE_WARNINGS_FILENAME = f"all_warnings_{DATE}.csv"
TEST_WARNINGS_FILENAME = f"test_warnings_{DATE}.csv"

CHECKPOINT_RESULTS_FILENAME = f"cp_results_{DATE}.csv"
CHECKPOINT_WARNINGS_FILENAME = f"cp_warnings_{DATE}.csv"

EXECUTION_REPORT_FILENAME = f"execution_report_{DATE}.json"
FORMAT_WARNINGS_FILENAME = f"format_warnings_{DATE}.csv"
ANNOTATED_RESULTS_FILENAME = f"annotated_results_{DATE}.csv"

CM_PARENT_PATH = f"cm_{DATE}"  # folder for classification matrices
CR_PARENT_PATH = f"cr_{DATE}"  # folder for classification reports

RESULTS_PARENT_PATH = "results"
METRICS_PARENT_PATH = "metrics"

COMPLETE_RESULTS_FILE = os.path.join(
    RESULTS_PARENT_PATH, MODEL_ABBR, COMPLETE_RESULTS_FILENAME
)
TEST_RESULTS_FILE = os.path.join(RESULTS_PARENT_PATH, MODEL_ABBR, TEST_RESULTS_FILENAME)

COMPLETE_WARNINGS_FILE = os.path.join(
    RESULTS_PARENT_PATH, MODEL_ABBR, COMPLETE_WARNINGS_FILENAME
)
TEST_WARNINGS_FILE = os.path.join(
    RESULTS_PARENT_PATH, MODEL_ABBR, TEST_WARNINGS_FILENAME
)

CHECKPOINT_RESULTS_FILE = os.path.join(
    RESULTS_PARENT_PATH, MODEL_ABBR, CHECKPOINT_RESULTS_FILENAME
)
CHECKPOINT_WARNINGS_FILE = os.path.join(
    RESULTS_PARENT_PATH, MODEL_ABBR, CHECKPOINT_WARNINGS_FILENAME
)

EXECUTION_REPORT_FILE = os.path.join(
    RESULTS_PARENT_PATH, MODEL_ABBR, EXECUTION_REPORT_FILENAME
)

FORMAT_WARNINGS_FILE = os.path.join(
    RESULTS_PARENT_PATH, MODEL_ABBR, FORMAT_WARNINGS_FILENAME
)

ANNOTATED_RESULTS_FILE = os.path.join(
    RESULTS_PARENT_PATH, MODEL_ABBR, ANNOTATED_RESULTS_FILENAME
)

CM_FOLDER = os.path.join(
    RESULTS_PARENT_PATH, MODEL_ABBR, METRICS_PARENT_PATH, CM_PARENT_PATH
)

CR_FOLDER = os.path.join(
    RESULTS_PARENT_PATH, MODEL_ABBR, METRICS_PARENT_PATH, CR_PARENT_PATH
)

FINETUNED_MODELS = "fine_tuned_models"

FINETUNED_MODEL_DIR = os.path.join(FINETUNED_MODELS, MODEL_ABBR)

LOGGING_DIR = os.path.join("logs", MODEL_ABBR)


class Config:
    is_test_run = IS_TEST_RUN
    dataset_name = DATASET_NAME
    is_fine_tune = IS_FINE_TUNE
    results_file = TEST_RESULTS_FILE if is_test_run else COMPLETE_RESULTS_FILE
    warnings_file = TEST_WARNINGS_FILE if is_test_run else COMPLETE_WARNINGS_FILE
    checkpoint_results_file = CHECKPOINT_RESULTS_FILE
    checkpoint_warnings_file = CHECKPOINT_WARNINGS_FILE
    execution_report_file = EXECUTION_REPORT_FILE
    format_warnings_file = FORMAT_WARNINGS_FILE
    annotated_results_file = ANNOTATED_RESULTS_FILE
    cm_folder = CM_FOLDER
    cr_folder = CR_FOLDER
    model = MODEL
    model_abbr = MODEL_ABBR
    finetuned_model_dir = FINETUNED_MODEL_DIR
    logging_dir = LOGGING_DIR
    save_logits = SAVE_LOGITS
    random_seed_path = os.path.join("random_seed.txt")

    batch_size = BATCH_SIZE
    learning_rate = 2e-5
    temperature = 1
    top_p = 1
    num_train_epochs = 3
    weight_decay = 0.01
    logging_steps = 10
    save_steps = 500
    save_total_limit = 2
    evaluation_strategy = "epoch"
    inference_threads = INFERENCE_THREADS
    max_output_length = MAX_OUTPUT_LENGTH
    test_size = 20
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    distributed_strategy = "ddp"
    train_threads = TRAIN_THREADS
    finetune_language = FINETUNE_LANGUAGE
    evaluate_languages = EVALUATE_LANGUAGES
    preprocess_threads = PREPROCESS_THREADS

    # pattern_xnli = ("entailment", "contradiction", "neutral", "not inferenced")
    # re_pattern_xnli = r"\b(" + "|".join(pattern_xnli) + r")\b"


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python parameters.py <config_attribute>")
        sys.exit(1)

    attribute = sys.argv[1]
    config = Config()

    if hasattr(config, attribute):
        print(getattr(config, attribute))
    else:
        print(f"Unknown attribute: {attribute}")
        sys.exit(1)
