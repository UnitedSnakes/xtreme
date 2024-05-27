# File: predict.py
# Author: Shanglin Yang
# Contact: syang662@wisc.edu

import os
import time
from typing import Tuple, Dict, Any
from transformers import (
    T5ForConditionalGeneration,
    AutoModelForCausalLM,
    BertForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
import numpy as np
import pandas as pd
from datasets import load_dataset, DatasetDict, Dataset
import argparse
import torch as th
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from parameters import Config
from utils import Color, request_user_confirmation

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
            elif "mbert" in model_name:
                cls._model = BertForSequenceClassification.from_pretrained(
                    model_name, device_map="auto"
                )
            else:
                cls._model = AutoModelForCausalLM.from_pretrained(
                    model_name, device_map="auto", torch_dtype=th.bfloat16
                )
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
            elif "mbert" in model_name:
                cls._model = BertForSequenceClassification.from_pretrained(output_dir)
            else:
                cls._model = AutoModelForCausalLM.from_pretrained(output_dir)
            cls._tokenizer = AutoTokenizer.from_pretrained(output_dir)
        return cls._model, cls._tokenizer


def load_dataset_and_preprocess(task_name, train=False):
    dataset = load_dataset(task_name)
    tokenizer = ModelLoader.get_tokenizer(Config.model)

    def preprocess_function(examples):
        inputs = tokenizer(
            examples["premise"], examples["hypothesis"], padding=True, truncation=False, max_length=tokenizer.model_max_length
        )
        # Check if input length exceeds the model's max length and log a warning
        too_long = []
        for i, input_ids in enumerate(inputs['input_ids']):
            if len(input_ids) > tokenizer.model_max_length:
                too_long.append(True)
            else:
                too_long.append(False)
        inputs["too_long"] = too_long
        return inputs

    # Initialize the "too_long" column with False
    dataset = dataset.map(lambda examples: {"too_long": [False] * len(examples["premise"])}, batched=True)
    encoded_dataset = dataset.map(preprocess_function, batched=True)

    if Config.is_test_run:
        encoded_dataset["test"] = encoded_dataset["test"].select(range(Config.test_size))
        if train:
            encoded_dataset["train"] = encoded_dataset["train"].select(range(Config.test_size))

    return encoded_dataset


def compute_metrics(eval_pred: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    acc = accuracy_score(labels, predictions)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def run_model_inference(model_name, dataset):
    # Filter out examples that are too long
    test_dataset = dataset["test"].filter(lambda example: not example["too_long"])
    
    trainer = Trainer(
        model=ModelLoader.get_model(model_name, inference_only=True),
        args=TrainingArguments(per_device_eval_batch_size=Config.batch_size),
        tokenizer=ModelLoader.get_tokenizer(model_name),
        compute_metrics=compute_metrics
    )
    predictions = trainer.predict(test_dataset)
    return predictions.predictions, predictions.metrics, test_dataset


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
    )

    # Filter out examples that are too long
    filtered_train_dataset = train_dataset.filter(lambda example: not example["too_long"])

    trainer = Trainer(
        model=ModelLoader.get_model(model_name, inference_only=False),
        args=training_args,
        train_dataset=filtered_train_dataset,
        tokenizer=ModelLoader.get_tokenizer(model_name),
        compute_metrics=compute_metrics
    )

    trainer.train()
    ModelLoader.save_model(Config.finetuned_model_dir)

def main(task_name, start_from_cp_file, overwrite):
    model_name = Config.model
    dataset = load_dataset_and_preprocess(task_name, train=Config.fine_tune)

    if Config.fine_tune:
        if not os.path.exists(Config.finetuned_model_dir):
            print("Fine-tuned model not found. Starting fine-tuning.")
            finetune_model(model_name, dataset["train"])
        else:
            print("Fine-tuned model found. Loading the fine-tuned model.")
            ModelLoader.load_finetuned_model(Config.finetuned_model_dir, model_name)

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
            
    # TODO: start from cp file
    
    predictions, metrics, test_dataset = run_model_inference(model_name, dataset)
    
    # Create a DataFrame for predictions and add the 'too_long' column
    df = pd.DataFrame(predictions)
    df["too_long"] = test_dataset["too_long"]
    
    df.to_csv(Config.results_file, index=False)
    print(f"Results stored in {Config.results_file}.")
    
    # Print metrics
    print("Metrics:", metrics)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run prediction model.")
    parser.add_argument("task_name", type=str, help="Task name for the dataset")
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
    parser.add_argument(
        "-overwrite",
        action="store_true",
        help="Overwrite existing results file.",
    )
    parser.add_argument(
        "-no_overwrite",
        action="store_false",
        dest="overwrite",
        help="Do not overwrite existing results file.",
    )
    args = parser.parse_args()

    t1 = time.time()
    main(args.task_name, args.start_from_cp_file, args.overwrite)
    t2 = time.time()
    print(f"Time taken: {t2 - t1:.2f}s")
