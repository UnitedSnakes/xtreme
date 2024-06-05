# File: parameters.py
# Author: Shanglin Yang
# Contact: syang662@wisc.edu

from datetime import datetime
import os
import torch as th

# --------------------modify below------------------------
# --------------------------------------------------------
# IS_TEST_RUN = True
IS_TEST_RUN = False
IS_FINE_TUNE = True
SAVE_LOGITS = False

# MODEL_ABBR = "FLAN_UL2"
# MODEL = "google/flan-ul2"

MODEL_ABBR = "MBERT"
MODEL = "bert-base-multilingual-cased"

BATCH_SIZE = 32
# GRAD_ACCUM_STEPS = 4
MAX_INPUT_LENGTH = 128
MAX_OUTPUT_LENGTH = 128
# steps_per_epoch = 392702 / 8 = 49088

# flan-ul2 = 40GB, so run 2 instances
INFERENCE_THREADS = 128
PREPROCESS_THREADS = 20
TRAIN_THREADS = 16
TEST_SIZE = 2

# https://huggingface.co/datasets/xnli
# DATASET_NAME = "xnli"
DATASET_NAME = "multi_nli"
# DATASET_NAME = "tatoeba"
SWAP_SENTENCES = False

# training hyperparameters
LEARNING_RATE = 2e-5
NUM_TRAIN_EPOCHS = 3
EVAL_STEPS = 100
SAVE_STEPS = 1000
LOGGING_STEPS = EVAL_STEPS
WARMUP_STEPS = 200

# --------------------modify above------------------------
# --------------------------------------------------------

TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')

LANG2_DICT = {
    "af": "afr",
    "ar": "ara",
    "bg": "bul",
    "bn": "ben",
    "de": "deu",
    "el": "ell",
    "es": "spa",
    "et": "est",
    "eu": "eus",
    "fa": "pes",
    "fi": "fin",
    "fr": "fra",
    "he": "heb",
    "hi": "hin",
    "hu": "hun",
    "id": "ind",
    "it": "ita",
    "ja": "jpn",
    "jv": "jav",
    "ka": "kat",
    "kk": "kaz",
    "ko": "kor",
    "ml": "mal",
    "mr": "mar",
    "nl": "nld",
    "pt": "por",
    "ru": "rus",
    "sw": "swh",
    "ta": "tam",
    "te": "tel",
    "th": "tha",
    "tl": "tgl",
    "tr": "tur",
    "ur": "urd",
    "vi": "vie",
    "zh": "cmn",
    "en": "eng",
    "az": "aze",
    "lt": "lit",
    "pl": "pol",
    "uk": "ukr",
    "ro": "ron",
}

if DATASET_NAME == "xnli":
    FINETUNE_LANGUAGE = "en"
    EVALUATE_LANGUAGES = (
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
    )
    PATTERNS = (
        ("0", "zero", "entailment"),
        ("1", "one", "neutral"),
        ("2", "two", "contradiction"),
        ("not inferenced"),
    )

elif DATASET_NAME == "tatoeba":
    FINETUNE_LANGUAGE = ""
    EVALUATE_LANGUAGES = "ar,he,vi,id,jv,tl,eu,ml,ta,te,af,nl,de,el,bn,hi,mr,ur,fa,fr,it,pt,es,bg,ru,ja,ka,ko,th,sw,zh,kk,tr,et,fi,hu".split(
        ","
    )
    PATTERNS = ()
    
elif DATASET_NAME == "multi_nli":
    FINETUNE_LANGUAGE = "en"
    EVALUATE_LANGUAGES = "en"
    PATTERNS = ()
    
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

CM = f"cm_{DATE}"  # folder for classification matrices
CR = f"cr_{DATE}"  # folder for classification reports

RESULTS = "results"
METRICS = "metrics"

RAW = "raw"
ANNOTATED = "annotated"

COMPLETE_RESULTS_FILE = os.path.join(
    RESULTS, MODEL_ABBR, DATASET_NAME, RAW, COMPLETE_RESULTS_FILENAME
)
TEST_RESULTS_FILE = os.path.join(
    RESULTS, MODEL_ABBR, DATASET_NAME, RAW, TEST_RESULTS_FILENAME
)

COMPLETE_WARNINGS_FILE = os.path.join(
    RESULTS, MODEL_ABBR, DATASET_NAME, RAW, COMPLETE_WARNINGS_FILENAME
)
TEST_WARNINGS_FILE = os.path.join(
    RESULTS, MODEL_ABBR, DATASET_NAME, RAW, TEST_WARNINGS_FILENAME
)

# CHECKPOINT_RESULTS_FILE = os.path.join(
#     RESULTS, MODEL_ABBR, CHECKPOINT_RESULTS_FILENAME
# )
# CHECKPOINT_WARNINGS_FILE = os.path.join(
#     RESULTS, MODEL_ABBR, CHECKPOINT_WARNINGS_FILENAME
# )

EXECUTION_REPORT_FILE = os.path.join(
    RESULTS, MODEL_ABBR, DATASET_NAME, RAW, EXECUTION_REPORT_FILENAME
)

# FORMAT_WARNINGS_FILE = os.path.join(
#     RESULTS, MODEL_ABBR, FORMAT_WARNINGS_FILENAME
# )

ANNOTATED_RESULTS_FILE = os.path.join(
    RESULTS, MODEL_ABBR, DATASET_NAME, ANNOTATED, ANNOTATED_RESULTS_FILENAME
)

METRICS_DIR = os.path.join(RESULTS, MODEL_ABBR, DATASET_NAME, METRICS)

CM_DIR = os.path.join(RESULTS, MODEL_ABBR, DATASET_NAME, METRICS, CM)

CR_DIR = os.path.join(RESULTS, MODEL_ABBR, DATASET_NAME, METRICS, CR)

FINETUNED_MODELS = "fine_tuned_models"

FINETUNED_MODEL_DIR = os.path.join(FINETUNED_MODELS, MODEL_ABBR, TIMESTAMP)

LOGGING_DIR = os.path.join("logs", MODEL_ABBR)

DEVICE = th.device("cuda" if th.cuda.is_available() else "cpu")

class Config:
    is_test_run = IS_TEST_RUN
    dataset_name = DATASET_NAME
    is_fine_tune = IS_FINE_TUNE
    results_file = TEST_RESULTS_FILE if is_test_run else COMPLETE_RESULTS_FILE
    warnings_file = TEST_WARNINGS_FILE if is_test_run else COMPLETE_WARNINGS_FILE
    # checkpoint_results_file = CHECKPOINT_RESULTS_FILE
    # checkpoint_warnings_file = CHECKPOINT_WARNINGS_FILE
    execution_report_file = EXECUTION_REPORT_FILE
    # format_warnings_file = FORMAT_WARNINGS_FILE
    annotated_results_file = ANNOTATED_RESULTS_FILE
    metrics_dir = METRICS_DIR
    cm_dir = CM_DIR
    cr_dir = CR_DIR
    model = MODEL
    model_abbr = MODEL_ABBR
    finetuned_model_dir = FINETUNED_MODEL_DIR
    logging_dir = LOGGING_DIR
    save_logits = SAVE_LOGITS
    random_seed_path = os.path.join("random_seed.txt")
    swap_sentences = SWAP_SENTENCES

    batch_size = BATCH_SIZE
    # grad_accum_steps = GRAD_ACCUM_STEPS
    learning_rate = LEARNING_RATE
    temperature = 1
    top_p = 1
    num_train_epochs = NUM_TRAIN_EPOCHS
    weight_decay = 0.01
    logging_steps = LOGGING_STEPS
    save_steps = SAVE_STEPS
    save_total_limit = 3
    eval_strategy = "steps"
    save_strategy = "steps"
    inference_threads = INFERENCE_THREADS
    max_output_length = MAX_OUTPUT_LENGTH
    max_input_length = MAX_INPUT_LENGTH
    test_size = TEST_SIZE
    device = DEVICE
    distributed_strategy = "ddp"
    train_threads = TRAIN_THREADS
    finetune_language = FINETUNE_LANGUAGE
    evaluate_languages = EVALUATE_LANGUAGES
    preprocess_threads = PREPROCESS_THREADS
    # early_stopping_patience = 3
    eval_steps = EVAL_STEPS
    warmup_steps = WARMUP_STEPS

    # 0 for entailment, 1 for neutral, 2 for contradiction
    patterns = PATTERNS
    lang2_dict = LANG2_DICT
    # re_pattern = RE_PATTERN
    
    timestamp = TIMESTAMP

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
