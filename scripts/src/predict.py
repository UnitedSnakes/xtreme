# File: predict.py
# Author: Shanglin Yang
# Contact: syang662@wisc.edu

import os
import time
import random
from typing import Tuple, Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import (
    T5ForConditionalGeneration,
    BertForSequenceClassification,
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

from parameters import Config
from utils import Color, request_user_confirmation, ensure_directory_exists_for_file

os.environ["HF_DATASETS_CACHE"] = "./cache"


class ModelLoader:
    _model = None
    _tokenizer = None

    @classmethod
    def get_model(cls, model_name, inference_only):
        if cls._model is None:
            if "flan-ul2" in model_name:
                cls._model = T5ForConditionalGeneration.from_pretrained(
                    model_name, device_map="auto", torch_dtype=th.bfloat16
                )
            elif "bert" in model_name:
                cls._model = BertForSequenceClassification.from_pretrained(model_name)
            else:
                raise NotImplementedError(f"Model {model_name} not supported.")
            if inference_only:
                cls._model.eval()
        return cls._model

    @classmethod
    def get_tokenizer(cls, model_name):
        if cls._tokenizer is None:
            cls._tokenizer = AutoTokenizer.from_pretrained(model_name)
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
                cls._model = BertForSequenceClassification.from_pretrained(output_dir)
            else:
                raise NotImplementedError(f"Model {model_name} not supported.")
            cls._tokenizer = AutoTokenizer.from_pretrained(output_dir)
        return cls._model, cls._tokenizer


def generate_prompt(examples, dataset_name: str):
    if dataset_name == "xnli":
        prompts = [
            (
                "- 'Entailment' means that Sentence2 logically follows from Sentence1.\n"
                "- 'Contradiction' means that Sentence2 logically conflicts with Sentence1.\n"
                "- 'Neutral' means that Sentence2 and Sentence1 are neither entailed nor contradictory.\n"
                "What is the relationship between Sentence1 and Sentence2? "
                f"Sentence1: {premise}\n\n"
                f"Sentence2: {hypothesis}\n\n"
                "Please answer with an exact label from the following: "
                "'entailment', 'contradiction', 'neutral'.\n\n"
            )
            for premise, hypothesis in zip(examples["premise"], examples["hypothesis"])
        ]
        return prompts
    # Add more cases for different datasets
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def load_dataset_by_name(
    task_name: str, language: str, split: str = None
) -> DatasetDict:
    if split:
        return load_dataset(task_name, language, split=split)
    return load_dataset(task_name, language)


def preprocess_dataset(
    dataset: DatasetDict, task_name: str, tokenizer, max_length: int
) -> DatasetDict:
    def preprocess_function(examples):
        prompts = generate_prompt(examples, task_name)
        inputs = tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        inputs["too_long"] = [
            len(input_ids) > max_length for input_ids in inputs["input_ids"]
        ]
        inputs["prompt"] = prompts
        return inputs

    return dataset.map(
        preprocess_function,
        batched=True,
        num_proc=Config.preprocess_threads,
        load_from_cache_file=True,
    )


def load_and_preprocess_dataset(
    task_name: str, language: str, train: bool = False
) -> DatasetDict:
    split = "train" if train else "test"
    dataset = load_dataset_by_name(task_name, language, split=split)
    tokenizer = ModelLoader.get_tokenizer(Config.model)

    if Config.is_test_run:
        dataset = dataset.select(range(Config.test_size))

    return preprocess_dataset(dataset, task_name, tokenizer, Config.model_max_length)


def compute_metrics(eval_pred: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="weighted"
    )
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


def run_inference_batch(model, tokenizer, inputs):
    with th.no_grad():
        if "bert" in Config.model:
            outputs = model(**inputs)
            predictions = th.argmax(outputs.logits, dim=-1)
        elif "flan-ul2" in Config.model:
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=Config.max_output_tokens,
                do_sample=True,
                temperature=Config.temperature,
            )
            predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return predictions, None


def run_model_inference(model_name, dataset):
    # Filter out examples that are too long
    valid_indices = [
        i for i, example in enumerate(dataset["test"]) if not example["too_long"]
    ]
    test_dataset = dataset["test"].select(valid_indices)

    model = ModelLoader.get_model(model_name, inference_only=True)
    tokenizer = ModelLoader.get_tokenizer(model_name)

    results = []
    errors = []

    with ThreadPoolExecutor(max_workers=Config.inference_threads) as executor:
        futures = [
            executor.submit(
                run_inference_batch,
                model,
                tokenizer,
                {
                    "input_ids": th.tensor(test_dataset[i]["input_ids"])
                    .unsqueeze(0)
                    .to(Config.device),
                    "attention_mask": th.tensor(test_dataset[i]["attention_mask"])
                    .unsqueeze(0)
                    .to(Config.device),
                },
            )
            for i in range(len(test_dataset))
        ]

        for future in as_completed(futures):
            prediction, error = future.result()
            if error:
                results.append("not inferenced: error: " + error)
                errors.append(error)
            else:
                results.append(prediction)
                errors.append(None)

    metrics = {}  # Replace with appropriate metrics calculation if needed

    return results, metrics, valid_indices, errors


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

    # Filter out examples that are too long
    valid_indices = [
        i for i, example in enumerate(train_dataset) if not example["too_long"]
    ]
    filtered_train_dataset = train_dataset.select(valid_indices)

    trainer = Trainer(
        model=ModelLoader.get_model(model_name, inference_only=False),
        args=training_args,
        train_dataset=filtered_train_dataset,
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


def main(start_from_cp_file, overwrite):
    model_name = Config.model

    # Few-shot fine-tuning
    if Config.fine_tune:
        seed = int(time.time())

        train_dataset = load_and_preprocess_dataset(
            Config.dataset_name, Config.finetune_language, train=True
        )

        few_shot_train_dataset = select_few_shot_samples(
            train_dataset["train"], Config.finetune_size, seed
        )

        with open(Config.random_seed_path, "w") as f:
            f.write(str(seed))

        if not os.path.exists(Config.finetuned_model_dir):
            print("Fine-tuned model not found. Starting fine-tuning.")
            finetune_model(model_name, few_shot_train_dataset)
        else:
            print("Fine-tuned model found. Loading the fine-tuned model.")
            ModelLoader.load_finetuned_model(Config.finetuned_model_dir, model_name)

    test_dataset = load_and_preprocess_dataset(
        Config.dataset_name, Config.evaluate_language, train=False
    )

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

    results, metrics, valid_indices, errors = run_model_inference(
        model_name, test_dataset
    )

    data = {
        "id": [i for i in range(len(test_dataset["test"]))],
        "prompt": [example["prompt"] for example in test_dataset["test"]],
        "prediction": [
            (
                "not inferenced: exceeding model input tokens limit"
                if i not in valid_indices
                else results[valid_indices.index(i)]
            )
            for i in range(len(test_dataset["test"]))
        ],
        "label": [example["label"] for example in test_dataset["test"]],
        "language": [Config.evaluate_language for example in test_dataset["test"]],
        "error": [
            "" if error is None else f"not inferenced: error: {error}"
            for error in errors
        ],
    }

    df = pd.DataFrame(data)

    ensure_directory_exists_for_file(Config.results_file)
    df.to_csv(Config.results_file, index=False)
    print(f"Results stored in {Config.results_file}.")


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

    t1 = time.time()
    main(args.start_from_cp_file, args.overwrite)
    t2 = time.time()
    print(f"Time taken: {t2 - t1:.2f}s")
