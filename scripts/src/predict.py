# File: predict.py
# Author: Shanglin Yang
# Contact: syang662@wisc.edu

import json
import os
import time
import random
from typing import Tuple, Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import (
    T5ForConditionalGeneration,
    BertTokenizer,
    BertModel,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)
import numpy as np
import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict
import argparse
import torch as th
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
from accelerate import Accelerator
import torch.nn.functional as F

from parameters import Config
from utils import (
    Color,
    load_dataset_by_name,
    request_user_confirmation,
    ensure_directory_exists_for_file,
    split_dict_in_half,
)

os.environ["HF_DATASETS_CACHE"] = "./cache"


class ModelLoader:
    _model = None
    _tokenizer = None
    _max_input_length = None

    @classmethod
    def get_model(cls, model_name, inference_only):
        if cls._model is None:
            if "flan-ul2" in model_name:
                cls._model = T5ForConditionalGeneration.from_pretrained(
                    model_name, device_map="auto", torch_dtype=th.bfloat16
                )
            elif "bert" in model_name:
                cls._model = BertModel.from_pretrained(model_name)
            else:
                raise NotImplementedError(f"Model {model_name} not supported.")
            if inference_only:
                cls._model.eval()
        return cls._model  # TODO: reset model after fine-tuning

    @classmethod
    def get_tokenizer(cls, model_name):
        if cls._tokenizer is None:
            if "flan-ul2" in model_name:
                cls._tokenizer = AutoTokenizer.from_pretrained(model_name)
            elif "bert" in model_name:
                cls._tokenizer = BertTokenizer.from_pretrained(model_name)
            else:
                raise NotImplementedError(f"Model {model_name} not supported.")
        return cls._tokenizer

    @classmethod
    def save_model(cls, output_dir):
        if cls._model is None or cls._tokenizer is None:
            raise ValueError("Model and tokenizer must be loaded before saving.")
        cls._model.save_pretrained(output_dir)
        cls._tokenizer.save_pretrained(output_dir)

    @classmethod
    def load_finetuned_model(cls, output_dir, model_name):
        if cls._model is None or cls._tokenizer is None:
            if "flan-ul2" in model_name:
                cls._model = T5ForConditionalGeneration.from_pretrained(output_dir)
            elif "bert" in model_name:
                cls._model = BertModel.from_pretrained(output_dir)
            else:
                raise NotImplementedError(f"Model {model_name} not supported.")
            cls._tokenizer = AutoTokenizer.from_pretrained(output_dir)
        return cls._model, cls._tokenizer

    @classmethod
    def get_max_input_length(cls, model_name):
        if "flan-ul2" in model_name:
            return 2048
        elif "bert" in model_name:
            return 512
        else:
            raise NotImplementedError(f"Model {model_name} not supported.")


def generate_prompt(example):
    # for flan-ul2, not for mbert
    # idx = example["id"]
    if Config.dataset_name == "xnli":
        prompts = [
            "You are a multilinguist. Given two sentences, determine the relationship between them:\n\n"
            f"Sentence 1: {premise}\n"
            f"Sentence 2: {hypothesis}\n\n"
            "Options:\n"
            "0. 'Entailment' - Sentence 2 logically follows from Sentence 1.\n"
            "1. 'Neutral' - Sentence 2 is neither entailed nor contradictory to Sentence 1.\n\n"
            "2. 'Contradiction' - Sentence 2 logically conflicts with Sentence 1.\n"
            "What is the relationship? Please answer with '0', '1', or '2'.\n"
            for premise, hypothesis in zip(example["premise"], example["hypothesis"])
        ]
        # prompts = [f"I am {idx}. Please tell me which number I am:\n"]
        return prompts
    elif Config.dataset_name == "tatoeba":
        prompts = [
            f"Translate the following sentence to {example['target_language']} and do not add any other content:\n\n"
            f"{example['source_sentence']}\n\n"
            for example in example
        ]
        return prompts
    raise ValueError(f"Unsupported dataset: {Config.dataset_name}")


def preprocess_dataset(dataset: DatasetDict, max_length: int = 0) -> DatasetDict:
    if "flan" in Config.model:
        return dataset.map(
            lambda example: {"prompt": generate_prompt(example)},
            batched=True,
            num_proc=Config.preprocess_threads,
            load_from_cache_file=True,
        )
    # if "bert" in Config.model:
    #     return dataset.map(
    #         lambda example: {
    #             "tokenized_target": ModelLoader.get_tokenizer(Config.model)(
    #                 example["targetString"],
    #                 padding=True,
    #                 truncation=False,
    #                 return_tensors="pt",
    #             ).to(Config.device)
    #         },
    #         batched=True,
    #         num_proc=Config.preprocess_threads,
    #         load_from_cache_file=True,
    #     )
    return dataset, max_length  # for mbert


def load_and_preprocess_dataset(language: str, train: bool = False) -> DatasetDict:
    split = "train" if train else "test"
    dataset = load_dataset_by_name(
        language, split=split, switch_source_and_target=Config.swap_sentences
    )

    if Config.dataset_name == "tatoeba":
        tokenizer = ModelLoader.get_tokenizer(Config.model)
        all_sentences = [
            sentence
            for example in dataset
            for sentence in [example["source_sentence"], example["target_sentence"]]
        ]

        max_length = max(
            len(tokenizer(sentence, truncation=True).input_ids)
            for sentence in all_sentences
        )
        print(f"max length:{max_length}")

        return preprocess_dataset(dataset, max_length)

    return preprocess_dataset(dataset)


def compute_metrics(eval_pred: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="weighted"
    )
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


@th.no_grad()
def run_inference_batch(inputs, tokenize_only=False, **kwargs):
    model = ModelLoader.get_model(Config.model, inference_only=True)
    tokenizer = ModelLoader.get_tokenizer(Config.model)

    if "bert" in Config.model:
        if Config.dataset_name == "tatoeba":
            assert tokenize_only

            max_length = kwargs.get("max_length")

            if (
                inputs.input_ids.shape[0] * inputs.input_ids.shape[1]
                > ModelLoader.get_max_input_length(Config.model)
                and inputs.input_ids.shape[0] == 1
            ):
                sentence_embedding = th.zeros(1, 768, device=Config.device)
            else:
                sentence_outputs = model(**inputs)
                # sentence_embedding = sentence_outputs.last_hidden_state[:, 0, :]
                sentence_embedding = sentence_outputs.last_hidden_state.mean(dim=1)

            targets = kwargs.get("targets")

            if (
                targets.input_ids.shape[0] * targets.input_ids.shape[1]
                > ModelLoader.get_max_input_length(Config.model)
                and targets.input_ids.shape[0] == 1
            ):
                target_embedding = th.zeros(1, 768, device=Config.device)
            else:
                target_outputs = model(**targets)
                # target_embedding = target_outputs.last_hidden_state[:, 0, :]
                target_embedding = target_outputs.last_hidden_state.mean(dim=1)

            return sentence_embedding, target_embedding
        else:
            raise NotImplementedError(f"Dataset {Config.dataset_name} not supported.")

    elif "flan-ul2" in Config.model:
        assert not tokenize_only
        outputs = model.generate(
            input_ids=inputs.input_ids,
            max_length=Config.max_output_length,
            do_sample=True,
            temperature=Config.temperature,
        )
        predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        print(predictions)
        return predictions

    else:
        raise NotImplementedError(f"Model {Config.model} not supported.")


def process_batch_recursively(batch, language, max_length):
    tokenizer = ModelLoader.get_tokenizer(Config.model)

    if "flan" in Config.model:
        if Config.dataset_name == "tatoeba":
            pass

        inputs = tokenizer(
            batch["prompt"], padding=True, truncation=False, return_tensors="pt"
        ).to(Config.device)
        input_length = inputs.input_ids.shape[0] * inputs.input_ids.shape[1]

    elif "bert" in Config.model:
        if Config.dataset_name == "tatoeba":
            try:
                inputs = tokenizer(
                    batch["source_sentence"],
                    padding="max_length",
                    max_length=max_length,
                    truncation=False,
                    return_tensors="pt",
                ).to(Config.device)
                targets = tokenizer(
                    batch["target_sentence"],
                    padding="max_length",
                    max_length=max_length,
                    truncation=False,
                    return_tensors="pt",
                ).to(Config.device)
            except Exception as e:
                # errorneous dataset
                return th.zeros(1, max_length, device=Config.device), th.zeros(
                    1, max_length, device=Config.device
                )
            input_length = max(
                inputs.input_ids.shape[0] * inputs.input_ids.shape[1],
                targets.input_ids.shape[0] * targets.input_ids.shape[1],
            )
        else:
            raise NotImplementedError(f"Dataset {Config.dataset_name} not supported.")
    else:
        raise NotImplementedError(f"Model {Config.model} not supported.")

    if input_length > ModelLoader.get_max_input_length(Config.model):
        # base case
        if len(batch["id"]) == 1:
            if Config.dataset_name == "tatoeba":
                return run_inference_batch(
                    inputs, tokenize_only=True, targets=targets, max_length=max_length
                )
            elif Config.dataset_name == "xnli":
                return [
                    f"not inferenced: input exceeded {Config.model}'s input token limits ({input_length} > {ModelLoader.get_max_input_length(Config.model)})"
                ]

        # Recursive case: split the batch into two halves
        first_half, second_half = split_dict_in_half(batch)
        # if "flan" in Config.model:
        if Config.dataset_name == "tatoeba":
            sentence_embedding1, target_embedding1 = process_batch_recursively(
                first_half, language, max_length
            )
            sentence_embedding2, target_embedding2 = process_batch_recursively(
                second_half, language, max_length
            )
            return th.vstack((sentence_embedding1, sentence_embedding2)), th.vstack(
                (target_embedding1, target_embedding2)
            )
        elif Config.dataset_name == "xnli":
            predictions1 = process_batch_recursively(first_half, language, max_length)
            predictions2 = process_batch_recursively(second_half, language, max_length)
            return predictions1 + predictions2
        else:
            raise NotImplementedError(f"Dataset {Config.dataset_name} not supported.")

    else:
        # Process the batch as is
        if "flan" in Config.model:
            return run_inference_batch(inputs, max_length=max_length)
        elif "bert" in Config.model:
            if Config.dataset_name == "tatoeba":
                return run_inference_batch(
                    inputs, tokenize_only=True, targets=targets, max_length=max_length
                )
            else:
                raise NotImplementedError(
                    f"Dataset {Config.dataset_name} not supported."
                )


def run_model_inference(dataset, language, max_length):
    # Initialize the accelerator
    accelerator = Accelerator()

    model = ModelLoader.get_model(Config.model, inference_only=True)
    tokenizer = ModelLoader.get_tokenizer(Config.model)

    # Prepare the model and tokenizer for inference
    model, tokenizer = accelerator.prepare(model, tokenizer)

    results = []
    errors = []

    batches = [
        dataset[i : i + Config.batch_size]
        for i in range(0, len(dataset), Config.batch_size)
    ]

    if Config.dataset_name == "tatoeba":
        sentence_embeddings = []
        target_embeddings = []

    with ThreadPoolExecutor(max_workers=Config.inference_threads) as executor:
        futures = {
            executor.submit(
                process_batch_recursively, batch, language, max_length
            ): batch
            for batch in batches
        }

        for future in as_completed(futures):
            batch = futures[future]
            # try:
            if Config.dataset_name == "xnli":
                predictions = future.result()
                results.extend(predictions)
                errors.extend([None] * len(batch["id"]))
            # except Exception as e:
            #     results.extend([f"not inferenced: error: {str(e)}"] * len(batch["prompt"]))
            #     errors.extend([str(e)] * len(batch["prompt"]))
            elif Config.dataset_name == "tatoeba":
                sentence_embedding, target_embedding = future.result()
                sentence_embeddings.append(sentence_embedding)
                target_embeddings.append(target_embedding)

                errors.extend([None] * len(batch["id"]))

    if Config.dataset_name == "tatoeba":
        sentence_embeddings = th.cat(sentence_embeddings, dim=0).to(Config.device)
        target_embeddings = th.cat(target_embeddings, dim=0).to(Config.device)

        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        target_embeddings = F.normalize(target_embeddings, p=2, dim=1)

        similarity_matrix = th.mm(
            sentence_embeddings, target_embeddings.transpose(0, 1)
        )
        max_indices = th.argmax(similarity_matrix, dim=1)
        true_indices = range(len(max_indices))
        results = [int((x == y).item()) for x, y in zip(max_indices, true_indices)]

        sentence_zero_rows = list((sentence_embeddings == 0).all(dim=1))
        target_zero_rows = list((target_embeddings == 0).all(dim=1))

        results = [
            -1 if flag else val for val, flag in zip(results, sentence_zero_rows)
        ]
        results = [-1 if flag else val for val, flag in zip(results, target_zero_rows)]

        print(results)

    return results, errors


class PrinterCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        print(logs)


def finetune_model(model_name, train_dataset):
    if not os.path.exists(Config.logging_dir):
        os.makedirs(Config.logging_dir)

    training_args = TrainingArguments(
        output_dir=Config.finetuned_model_dir,
        evaluation_strategy=Config.evaluation_strategy,
        learning_rate=Config.learning_rate,
        per_device_train_batch_size=Config.batch_size,
        per_device_eval_batch_size=Config.batch_size,
        num_train_epochs=Config.num_train_epochs,
        weight_decay=Config.weight_decay,
        logging_dir=Config.logging_dir,
        logging_steps=Config.logging_steps,
        save_steps=Config.save_steps,
        save_total_limit=Config.save_total_limit,
        # Distributed training arguments
        dataloader_num_workers=Config.train_threads,
        ddp_find_unused_parameters=False,
        # Distributed strategy
        distributed_strategy=Config.distributed_strategy,
    )

    trainer = Trainer(
        model=ModelLoader.get_model(model_name, inference_only=False),
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=ModelLoader.get_tokenizer(model_name),
        compute_metrics=compute_metrics,
        callbacks=[PrinterCallback],
    )

    trainer.train()
    ModelLoader.save_model(Config.finetuned_model_dir)

    # Save training loss plot
    plot_and_save_loss(trainer, Config.logging_dir)


def plot_and_save_loss(trainer, output_dir):
    # Get training log history
    history = trainer.state.log_history

    # Extract loss values
    train_loss = [entry["loss"] for entry in history if "loss" in entry]
    eval_loss = [entry["eval_loss"] for entry in history if "eval_loss" in entry]

    # Plot training and evaluation loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss, label="Training Loss")
    plt.plot(eval_loss, label="Evaluation Loss")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Evaluation Loss over Time")

    # Save the plot
    plt_path = os.path.join(output_dir, "training_loss.png")
    plt.savefig(plt_path)
    print(f"Training loss plot saved at {plt_path}")


def select_few_shot_samples(dataset: Dataset, size: int, seed: int) -> Dataset:
    # Set random seed for reproducibility
    np.random.seed(seed)
    selected_indices = np.random.choice(len(dataset), size, replace=False)
    return dataset.select(selected_indices)


def save_running_logs(df_with_errors, time_, language):
    config_instance = Config()

    report_dict = {
        attr: getattr(config_instance, attr)
        for attr in dir(config_instance)
        if not attr.startswith("__") and not callable(getattr(config_instance, attr))
    }

    report_dict["device"] = str(report_dict["device"])

    errors_list_dict = df_with_errors.to_dict("records")

    report_dict["df_with_errors"] = errors_list_dict
    report_dict["time"] = time_

    del report_dict["lang2_dict"]

    with open(
        Config.execution_report_file.replace(".json", f"_{language}.json"), "w"
    ) as json_file:
        json.dump(report_dict, json_file, indent=4)

    print("\n-------------------error messages-------------------")
    print(df_with_errors, end="\n\n")

    print("-----------------time-----------------")
    print(f"time: {time_:.2f}s")


def main(start_from_cp_file, overwrite):
    # Few-shot fine-tuning
    if Config.is_fine_tune:
        with open(Config.random_seed_path, "r") as f:
            seed = int(f.read())

        train_dataset, _ = load_and_preprocess_dataset(
            Config.finetune_language, train=True
        )

        few_shot_train_dataset = select_few_shot_samples(
            train_dataset["train"], Config.finetune_size, seed
        )

        with open(Config.random_seed_path, "w") as f:
            f.write(str(seed))

        if not os.path.exists(Config.finetuned_model_dir):
            print("Fine-tuned model not found. Starting fine-tuning.")
            finetune_model(Config.model, few_shot_train_dataset)
        else:
            print("Fine-tuned model found. Loading the fine-tuned model.")
            ModelLoader.load_finetuned_model(Config.finetuned_model_dir, Config.model)

    # i = 0
    # flag = True

    for language in Config.evaluate_languages:
        # if i < 32:
        #     i += 1
        #     continue
        # if language != "tr" and flag:
        # continue
        # else:
        # flag = False
        print(f"Processing {language}...")
        t1 = time.time()

        try:
            test_dataset, max_length = load_and_preprocess_dataset(
                language, train=False
            )
        except ValueError as e:
            print(f"Error: {e}")
            continue

        if overwrite is False and os.path.exists(Config.results_file):
            print(
                f"{Config.results_file} already exists. Not allowed to overwrite. Skipping inference."
            )
            exit(0)
        elif overwrite is None and os.path.exists(Config.results_file):
            print(f"{Config.results_file} already exists.\n")
            if not request_user_confirmation(
                f"{Color.YELLOW}Overwrite?{Color.END} [Y/n]: ",
                "Overwriting...",
                f"{Config.results_file} is skipped.",
                auto_confirm="y" if overwrite else None,
            ):
                exit(0)

        results, errors = run_model_inference(test_dataset, language, max_length)

        if "flan" in Config.model:
            data = {
                "id": test_dataset["id"],
                "prompt": [example["prompt"] for example in test_dataset],
                "prediction": results,
                "label": [example["label"] for example in test_dataset],
                "language": [language for example in test_dataset],
                "error": errors,
            }

        elif "bert" in Config.model:
            if Config.dataset_name == "xnli":
                data = {
                    "id": test_dataset["id"],
                    "prompt": [example["prompt"] for example in test_dataset],
                    "prediction": results,
                    "label": [example["label"] for example in test_dataset],
                    "language": [language for example in test_dataset],
                    "error": errors,
                }  # TODO
            elif Config.dataset_name == "tatoeba":
                data = {
                    "id": test_dataset["id"],
                    "prediction": results,
                    "language": [language for example in test_dataset],
                    "error": errors,
                }
            else:
                raise NotImplementedError(
                    f"Dataset {Config.dataset_name} not supported."
                )

        else:
            raise NotImplementedError(f"Model {Config.model} not supported.")

        # print(data)
        df = pd.DataFrame(data)

        ensure_directory_exists_for_file(Config.results_file)
        df.to_csv(Config.results_file.replace(".csv", f"_{language}.csv"), index=False)
        print(
            f"Results stored in {Config.results_file.replace('.csv', f'_{language}.csv')}."
        )

        df_with_errors = df[df["error"].notna()]

        if not df_with_errors.empty:
            ensure_directory_exists_for_file(Config.warnings_file)
            df_with_errors.to_csv(Config.warnings_file, index=False)
            print(
                f"WARNING: Rows with not empty 'notes' column saved to {Config.warnings_file}"
            )
        else:
            print("Good news! No rows with not empty 'notes' column found.")

        t2 = time.time()
        print(f"Processing {language} done.")
        print(f"Time taken: {t2 - t1:.2f}s")

        save_running_logs(df_with_errors, t2 - t1, language)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run prediction model.")

    parser.add_argument(
        "-start_from_cp_file",
        action="store_true",
        help="Start from checkpoint file if exists.",
    )
    parser.add_argument(
        "-no_start_from_cp_file",
        action="store_false",
        dest="start_from_cp_file",
        help="Do not start from checkpoint file.",
    )
    parser.set_defaults(start_from_cp_file=None)

    parser.add_argument(
        "-overwrite", action="store_true", help="Overwrite existing results file."
    )
    parser.add_argument(
        "-no_overwrite",
        action="store_false",
        dest="overwrite",
        help="Do not overwrite existing results file.",
    )
    parser.set_defaults(overwrite=None)

    args = parser.parse_args()

    main(args.start_from_cp_file, args.overwrite)
