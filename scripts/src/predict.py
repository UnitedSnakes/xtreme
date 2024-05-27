# File: predict.py
# Author: Shanglin Yang
# Contact: syang662@wisc.edu

import os
import time
from typing import Tuple, Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import (
    T5ForConditionalGeneration,
    BertForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
import numpy as np
import pandas as pd
from datasets import load_dataset
import argparse
import torch as th
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt

from parameters import Config
from utils import Color, request_user_confirmation, ensure_directory_exists_for_file

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
            elif "mbert" in model_name:
                cls._model = BertForSequenceClassification.from_pretrained(output_dir)
            else:
                raise NotImplementedError(f"Model {model_name} not supported.")
            cls._tokenizer = AutoTokenizer.from_pretrained(output_dir)
        return cls._model, cls._tokenizer


def generate_prompt(example: Dict[str, Any], dataset_name: str) -> str:
    if dataset_name == "xnli":
        return (f"Sentence1: {example['premise']}\n\n"
                f"Sentence2: {example['hypothesis']}\n\n"
                f"What is the relationship between Sentence1 and Sentence2? "
                "Please answer with an exact label from the following: "
                "'entailment', 'contradiction', 'neutral'.")
    # Add more conditions for different datasets
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

def load_dataset_and_preprocess(task_name, train=False):
    dataset = load_dataset(task_name)
    tokenizer = ModelLoader.get_tokenizer(Config.model)

    def preprocess_function(examples):
        prompts = [generate_prompt(ex, task_name) for ex in examples]
        inputs = tokenizer(prompts, padding=True, truncation=False, max_length=tokenizer.model_max_length)
        
        too_long = []
        for i, input_ids in enumerate(inputs['input_ids']):
            if len(input_ids) > tokenizer.model_max_length:
                inputs["input_ids"][i] = input_ids[:tokenizer.model_max_length]
                inputs["attention_mask"][i] = inputs["attention_mask"][i][:tokenizer.model_max_length]
                too_long.append(True)
            else:
                too_long.append(False)
        inputs["too_long"] = too_long
        inputs["prompt"] = prompts
        return inputs

    dataset = dataset.map(preprocess_function, batched=True)

    if Config.is_test_run:
        dataset["test"] = dataset["test"].select(range(Config.test_size))
        if train:
            dataset["train"] = dataset["train"].select(range(Config.test_size))

    return dataset


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

def run_inference_batch(model, tokenizer, inputs):
    try:
        outputs = model.generate(**inputs)
        return outputs, None
    except Exception as e:
        return None, str(e)

def run_model_inference(model_name, dataset):
    # Filter out examples that are too long
    valid_indices = [i for i, example in enumerate(dataset["test"]) if not example["too_long"]]
    test_dataset = dataset["test"].select(valid_indices)
    
    model = ModelLoader.get_model(model_name, inference_only=True)
    tokenizer = ModelLoader.get_tokenizer(model_name)
    
    results = []
    errors = []

    with ThreadPoolExecutor(max_workers=Config.inference_threads) as executor:
        futures = [executor.submit(run_inference_batch, model, tokenizer, {"input_ids": th.tensor(test_dataset[i]['input_ids']).unsqueeze(0).to(Config.device), "attention_mask": th.tensor(test_dataset[i]['attention_mask']).unsqueeze(0).to(Config.device)}) for i in range(len(test_dataset))]
        
        for future in as_completed(futures):
            output, error = future.result()
            if error:
                results.append("not inferenced: error: " + error)
                errors.append(error)
            else:
                prediction = tokenizer.decode(output[0], skip_special_tokens=True)
                results.append(prediction)
                errors.append(None)

    metrics = {}  # Replace with appropriate metrics calculation if needed

    return results, metrics, valid_indices, errors


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
        distributed_strategy=Config.distributed_strategy
    )

    # Filter out examples that are too long
    valid_indices = [i for i, example in enumerate(train_dataset) if not example["too_long"]]
    filtered_train_dataset = train_dataset.select(valid_indices)

    trainer = Trainer(
        model=ModelLoader.get_model(model_name, inference_only=False),
        args=training_args,
        train_dataset=filtered_train_dataset,
        tokenizer=ModelLoader.get_tokenizer(model_name),
        compute_metrics=compute_metrics
    )

    trainer.train()
    ModelLoader.save_model(Config.finetuned_model_dir)
    
    # Save training loss plot
    plot_and_save_loss(trainer, Config.logging_dir)


def plot_and_save_loss(trainer, output_dir):
    # Get training log history
    history = trainer.state.log_history

    # Extract loss values
    train_loss = [entry['loss'] for entry in history if 'loss' in entry]
    eval_loss = [entry['eval_loss'] for entry in history if 'eval_loss' in entry]

    # Plot training and evaluation loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss, label='Training Loss')
    plt.plot(eval_loss, label='Evaluation Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Evaluation Loss over Time')
    
    # Save the plot
    plt_path = os.path.join(output_dir, 'training_loss.png')
    plt.savefig(plt_path)
    print(f"Training loss plot saved at {plt_path}")


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

    results, metrics, valid_indices, errors = run_model_inference(model_name, dataset)
    
    # Create a DataFrame for predictions
    data = {
        'prompt': [example["prompt"] for example in dataset["test"]],
        'prediction': ['not inferenced: exceeding model input tokens limit' if i not in valid_indices else results[valid_indices.index(i)] for i in range(len(dataset["test"]))],
        'label': [example["label"] for example in dataset["test"]],
        'id': [example["id"] for example in dataset["test"]],
        'language': [example["language"] for example in dataset["test"]],
        'error': ['' if error is None else f"not inferenced: error: {error}" for error in errors]
    }

    df = pd.DataFrame(data)
    
    df.to_csv(Config.results_file, index=False)
    print(f"Results stored in {Config.results_file}.")
    
    # Print metrics
    # print("Metrics:", metrics)

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
