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
from accelerate import Accelerator

from parameters import Config
from utils import Color, request_user_confirmation, ensure_directory_exists_for_file

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

    @classmethod
    def get_max_input_length(cls, model_name):
        if "flan-ul2" in model_name:
            return 2048
        elif "bert" in model_name:
            return 512
        else:
            raise NotImplementedError(f"Model {model_name} not supported.")


def generate_prompt(example, dataset_name: str):
    # idx = example["index"]
    if dataset_name == "xnli":
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
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def load_dataset_by_name(
    task_name: str, language: str, split: str = None
) -> DatasetDict:
    if split:
        return load_dataset(task_name, language, split=split)
    return load_dataset(task_name, language)


def preprocess_dataset(
    dataset: DatasetDict, task_name: str
) -> DatasetDict:
    def preprocess_function(example):
        prompts = generate_prompt(example, task_name)
        
        return {"prompt": prompts}
    
    # def add_index(example, idx):
    #     example["index"] = idx
    #     return example

    # # Add index to each example
    # dataset = dataset.map(add_index, with_indices=True, num_proc=Config.preprocess_threads)

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

    if Config.is_test_run:
        dataset = dataset.select(range(Config.test_size))

    return preprocess_dataset(dataset, task_name)


def compute_metrics(eval_pred: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="weighted"
    )
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

@th.no_grad()
def run_inference_batch(model, tokenizer, input_ids):
    if "bert" in Config.model:
        outputs = model(**inputs)
        predictions = th.argmax(outputs.logits, dim=-1)
    elif "flan-ul2" in Config.model:
        outputs = model.generate(
            input_ids=input_ids,
            max_length=Config.max_output_length,
            do_sample=True,
            temperature=Config.temperature,
        )
        predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
    print(predictions)
    return predictions


def batch_inference(model, tokenizer, input_ids):
    # inputs = {
    #     "input_ids": th.stack([th.tensor(ex) for ex in batch["input_ids"]]),
    #     "attention_mask": th.stack([th.tensor(ex) for ex in batch["attention_mask"]])
    # }

    predictions = run_inference_batch(model, tokenizer, input_ids)
    return predictions


def process_batch_recursively(prompts, model, tokenizer):
    if len(prompts) == 1:
        # Base case: only one input in the batch
        input_ids = ModelLoader.get_tokenizer(Config.model)(
            prompts, return_tensors="pt"
        ).input_ids.to(Config.device)
        input_length = input_ids.shape[1]
        if input_length > ModelLoader.get_max_input_length():
            return [f"not inferenced: input exceeded {Config.model}'s input token limits ({input_length} > {ModelLoader.get_max_input_length()})"]
        else:
            return batch_inference(model, tokenizer, input_ids)
    
    input_ids = ModelLoader.get_tokenizer(Config.model)(
        prompts, padding=True,return_tensors="pt"
    ).input_ids.to(Config.device)
    
    input_length = input_ids.shape[0] * input_ids.shape[1]
    if input_length > ModelLoader.get_max_input_length(Config.model):
        # Recursive case: split the batch into two halves
        
        mid = len(prompts) // 2
        prompts1 = prompts[:mid]
        prompts2 = prompts[mid:]

        predictions1 = process_batch_recursively(prompts1, model, tokenizer)
        predictions2 = process_batch_recursively(prompts2, model, tokenizer)
        
        return predictions1 + predictions2
    else:
        # Process the batch as is
        return batch_inference(model, tokenizer, input_ids)


def run_model_inference(model_name, dataset, batch_size):
    # Initialize the accelerator
    accelerator = Accelerator()

    model = ModelLoader.get_model(model_name, inference_only=True)
    tokenizer = ModelLoader.get_tokenizer(model_name)

    # Prepare the model and tokenizer for inference
    model, tokenizer = accelerator.prepare(model, tokenizer)

    results = []
    errors = []

    def process_batch(batch):
        return process_batch_recursively(batch["prompt"], model, tokenizer)

    batches = [dataset[i:i + batch_size] for i in range(0, len(dataset), batch_size)]

    with ThreadPoolExecutor(max_workers=Config.inference_threads) as executor:
        futures = {executor.submit(process_batch, batch): batch for batch in batches}

        for future in as_completed(futures):
            batch = futures[future]
            # print(batch)
            # print(type(batch))
            # try:
            predictions = future.result()
            results.extend(predictions)
            errors.extend([None] * len(batch["prompt"]))
            # except Exception as e:
            #     results.extend([f"not inferenced: error: {str(e)}"] * len(batch["prompt"]))
            #     errors.extend([str(e)] * len(batch["prompt"]))

    metrics = {}  # Replace with appropriate metrics calculation if needed

    return results, metrics, errors


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

    with open(Config.execution_report_file.replace(".json", f"_{language}.json"), "w") as json_file:
        json.dump(report_dict, json_file, indent=4)

    print("\n-------------------error messages-------------------")
    print(df_with_errors, end="\n\n")

    print("-----------------time-----------------")
    print(f"time: {time_:.2f}s")
    

def main(start_from_cp_file, overwrite):
    # Few-shot fine-tuning
    if Config.is_fine_tune:
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
            finetune_model(Config.model, few_shot_train_dataset)
        else:
            print("Fine-tuned model found. Loading the fine-tuned model.")
            ModelLoader.load_finetuned_model(Config.finetuned_model_dir, Config.model)

    for language in Config.evaluate_languages:
        t1 = time.time()
        
        test_dataset = load_and_preprocess_dataset(
            Config.dataset_name, language, train=False
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

        results, metrics, errors = run_model_inference(
            Config.model, test_dataset, batch_size=Config.batch_size
        )
        

        data = {
            "id": [i for i in range(len(test_dataset))],
            "prompt": [example["prompt"] for example in test_dataset],
            "prediction": results,
            "label": [example["label"] for example in test_dataset],
            "language": [language for example in test_dataset],
            "error": errors,
        }
        
        df = pd.DataFrame(data)

        ensure_directory_exists_for_file(Config.results_file)
        df.to_csv(Config.results_file.replace('.csv', f'_{language}.csv'), index=False)
        print(f"Results stored in {Config.results_file.replace('.csv', f'_{language}')}.")
        
        df_with_errors = df[df["error"].notna()]
        
        if not df_with_errors.empty:
            ensure_directory_exists_for_file(Config.warnings_file)
            df_with_errors.to_csv(Config.warnings_file, index=False)
            print(f"WARNING: Rows with not empty 'notes' column saved to {Config.warnings_file}")
        else:
            print("Good news! No rows with not empty 'notes' column found.")
        
        t2 = time.time()
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

    