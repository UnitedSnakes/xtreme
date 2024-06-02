# File: predict.py
# Author: Shanglin Yang
# Contact: syang662@wisc.edu

from datetime import datetime
import json
import os
import sys
import time
from typing import Tuple, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import (
    T5ForConditionalGeneration,
    BertTokenizer,
    # BertModel,
    BertForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    DataCollatorWithPadding,
    EarlyStoppingCallback
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
from torch import multiprocessing
from torch.utils.data import DataLoader
from tqdm import tqdm
import warnings
import logging

from parameters import Config
from utils import (
    Color,
    MultilineFormatter,
    PrintLogger,
    add_index_column,
    get_latest_checkpoint_dir,
    load_dataset_by_name,
    request_user_confirmation,
    ensure_directory_exists_for_file,
    split_dict_in_half,
    # ColorFormatter
)


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
                # cls._model = BertModel.from_pretrained(model_name)
                cls._model = BertForSequenceClassification.from_pretrained(
                    model_name, num_labels=3, output_hidden_states=True
                )
            else:
                raise NotImplementedError(f"Model {model_name} not supported.")
            if inference_only:
                cls._model.eval()
            # cls._model = th.nn.DataParallel(cls._model)
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
                # cls._model = BertModel.from_pretrained(output_dir)
                cls._model = BertForSequenceClassification.from_pretrained(
                    output_dir, num_labels=3, output_hidden_states=True
                )
            else:
                raise NotImplementedError(f"Model {model_name} not supported.")
            # cls._model = th.nn.DataParallel(cls._model)
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


def preprocess_dataset(dataset: DatasetDict, max_length: int) -> DatasetDict:
    if "flan" in Config.model:
        dataset = dataset.map(
            lambda ex: {"prompt": generate_prompt(ex)},
            batched=True,
            num_proc=Config.preprocess_threads,
            load_from_cache_file=True,
        )
    elif Config.dataset_name == "multi_nli":

        def preprocess_function(example):
            premise = example["premise"]
            hypothesis = example["hypothesis"]
            tokenized_input = ModelLoader.get_tokenizer(Config.model)(
                premise,
                hypothesis,
                padding="max_length",
                truncation=True,
                max_length=Config.max_input_length,
                return_tensors="pt",
            )

            input_ids = tokenized_input["input_ids"].squeeze(0).to(Config.device)
            attention_mask = tokenized_input["attention_mask"].squeeze(0).to(Config.device)

            labels = th.tensor(example["label"]).to(Config.device)
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }

        dataset = dataset.map(
            # lambda ex: {
            #     "input_ids_premise": ModelLoader.get_tokenizer(Config.model)(ex["premise"], padding=True, truncation=True)["input_ids"],
            #     "attention_mask_premise": ModelLoader.get_tokenizer(Config.model)(ex["premise"], padding=True, truncation=True)["attention_mask"],
            #     "input_ids_hypothesis": ModelLoader.get_tokenizer(Config.model)(ex["hypothesis"], padding=True, truncation=True)["input_ids"],
            #     "attention_mask_hypothesis": ModelLoader.get_tokenizer(Config.model)(ex["hypothesis"], padding=True, truncation=True)["attention_mask"],
            # },
            preprocess_function,
            # batched=True,
            # num_proc=Config.preprocess_threads,
            load_from_cache_file=False,
            remove_columns=[
                "promptID",
                "premise",
                "hypothesis",
                "premise_binary_parse",
                "premise_parse",
                "hypothesis_binary_parse",
                "hypothesis_parse",
                "genre",
                "pairID",
                "label"
            ],
        )
    else:
        raise NotImplementedError(f"Dataset {Config.dataset_name} not supported.")
    return dataset  # for mbert


def load_and_preprocess_dataset(
    language: str,
    split: str,
    add_index: bool = True,
) -> DatasetDict:
    dataset = load_dataset_by_name(
        language, split=split, switch_source_and_target=Config.swap_sentences
    )

    max_length = 0

    if Config.dataset_name == "multi_nli":
        tokenizer = ModelLoader.get_tokenizer(Config.model)
        all_sentences = [
            sentence
            for example in dataset
            for sentence in [example["premise"], example["hypothesis"]]
        ]

        max_length = max(
            len(tokenizer(sentence, truncation=True).input_ids)
            for sentence in all_sentences
        )
        print(f"max length:{max_length}")

    dataset = preprocess_dataset(dataset, max_length)

    if add_index and not Config.is_fine_tune:
        dataset = add_index_column(dataset)
        
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    return dataset

def compute_metrics(eval_pred: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
    outputs, labels = eval_pred
    logits, _ = outputs
    predictions = np.argmax(logits, axis=-1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="weighted", zero_division=0
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

            if (
                inputs.input_ids.shape[0] * inputs.input_ids.shape[1]
                > ModelLoader.get_max_input_length(Config.model)
                and inputs.input_ids.shape[0] == 1
            ):
                sentence_embedding = th.zeros(1, 768, device=Config.device)
            else:
                sentence_outputs = model(**inputs)
                # sentence_embedding = sentence_outputs.last_hidden_state[:, 0, :]
                # sentence_embedding = sentence_outputs.last_hidden_state.mean(dim=1)
                hidden_states = sentence_outputs.hidden_states[-1]
                sentence_embedding = hidden_states.mean(dim=1)

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
                # target_embedding = target_outputs.last_hidden_state.mean(dim=1)
                hidden_states = target_outputs.hidden_states[-1]
                target_embedding = hidden_states.mean(dim=1)

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
                return run_inference_batch(inputs, tokenize_only=True, targets=targets)
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
            return run_inference_batch(inputs)
        elif "bert" in Config.model:
            if Config.dataset_name == "tatoeba":
                return run_inference_batch(inputs, tokenize_only=True, targets=targets)
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
    def __init__(self, logging_steps):
        self.logging_steps = logging_steps
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        log_msg = f"\n\nStep {Color.YELLOW}{state.global_step}{Color.END}: "
        if 'eval_loss' in logs:
            log_msg += f"Eval - {Color.RED}Loss{Color.END}: {logs['eval_loss']}, {Color.GREEN}Accuracy{Color.END}: {logs['eval_accuracy']}\n"
            print(log_msg)
        elif 'train_loss' in logs:
            log_msg += f"Train - {Color.RED}Loss{Color.END}: {logs['train_loss']}\n"
            print(log_msg)
            
class SavePlotCallback(TrainerCallback):
    def __init__(self, logging_steps, trainer):
        self.logging_steps = logging_steps
        self.trainer = trainer
        self.previous_loss_plot_path = None
        self.previous_accuracy_plot_path = None

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.logging_steps == 0:
            loss_plot_path, accuracy_plot_path = plot_and_save_metrics(self.trainer, Config.logging_dir, state.global_step)
            
            if self.previous_loss_plot_path is not None:
                os.remove(self.previous_loss_plot_path)
            self.previous_loss_plot_path = loss_plot_path
            
            if self.previous_accuracy_plot_path is not None:
                os.remove(self.previous_accuracy_plot_path)
            self.previous_accuracy_plot_path = accuracy_plot_path


def finetune_model(model_name, train_dataloader, eval_dataloader, resume_from_checkpoint=None):
    if not os.path.exists(Config.logging_dir):
        os.makedirs(Config.logging_dir)
    if not os.path.exists(Config.finetuned_model_dir):
        os.makedirs(Config.finetuned_model_dir)
        
    training_args = TrainingArguments(
        output_dir=os.path.join(Config.finetuned_model_dir),
        eval_strategy=Config.eval_strategy,
        save_strategy=Config.save_strategy,
        learning_rate=Config.learning_rate,
        per_device_train_batch_size=Config.batch_size,
        per_device_eval_batch_size=Config.batch_size,
        num_train_epochs=Config.num_train_epochs,
        weight_decay=Config.weight_decay,
        logging_dir=Config.logging_dir,
        logging_steps=Config.logging_steps,
        save_steps=Config.save_steps,
        eval_steps=Config.eval_steps,
        save_total_limit=Config.save_total_limit,
        # Distributed training arguments
        dataloader_num_workers=Config.train_threads,
        ddp_find_unused_parameters=False,
        # # Distributed strategy
        # distributed_strategy=Config.distributed_strategy,
        fp16=True,
        resume_from_checkpoint=resume_from_checkpoint,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False
    )
    
    th.cuda.empty_cache()

    trainer = Trainer(
        model=ModelLoader.get_model(model_name, inference_only=False),
        args=training_args,
        train_dataset=train_dataloader.dataset,
        eval_dataset=eval_dataloader.dataset,
        tokenizer=ModelLoader.get_tokenizer(model_name),
        compute_metrics=compute_metrics,
        callbacks=[
            PrinterCallback(logging_steps=Config.logging_steps),
            EarlyStoppingCallback(early_stopping_patience=15 * Config.eval_steps),
        ],
    )
    
    trainer.add_callback(SavePlotCallback(logging_steps=Config.logging_steps, trainer=trainer))

    trainer.train()
    ModelLoader.save_model(Config.finetuned_model_dir)

    # Save training loss plot
    plot_and_save_metrics(trainer, Config.logging_dir, "final")
    
    # Log training summary
    # print(f"{Color.YELLOW}Training summary{Color.END}:")
    # print(trainer.state.log_history)

def plot_and_save_metrics(trainer, output_dir, step):
    # Get training log history
    history = trainer.state.log_history

    # Extract loss values
    train_loss = [entry["loss"] for entry in history if "loss" in entry]
    eval_loss = [entry["eval_loss"] for entry in history if "eval_loss" in entry]

    # Extract accuracy values
    train_accuracy = [entry["train_accuracy"] for entry in history if "train_accuracy" in entry]
    eval_accuracy = [entry["eval_accuracy"] for entry in history if "eval_accuracy" in entry]

    # Plot training and evaluation loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss, label="Training Loss")
    plt.plot(eval_loss, label="Evaluation Loss")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Evaluation Loss over Time")

    # Save the loss plot
    loss_plot_path = os.path.join(output_dir, f"training_loss_step_{step}_{Config.timestamp}.png")
    plt.savefig(loss_plot_path)
    plt.close()
    print(f"\nTraining loss plot saved at {loss_plot_path}")

    # Plot training and evaluation accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(train_accuracy, label="Training Accuracy")
    plt.plot(eval_accuracy, label="Evaluation Accuracy")
    plt.xlabel("Steps")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Training and Evaluation Accuracy over Time")

    # Save the accuracy plot
    accuracy_plot_path = os.path.join(output_dir, f"training_accuracy_step_{step}_{Config.timestamp}.png")
    plt.savefig(accuracy_plot_path)
    plt.close()
    print(f"\nTraining accuracy plot saved at {accuracy_plot_path}")

    return loss_plot_path, accuracy_plot_path


def select_few_shot_samples(dataset: Dataset, size: int, seed: int) -> Dataset:
    # Set random seed for reproducibility
    np.random.seed(seed)
    selected_indices = np.random.choice(len(dataset), size, replace=False)
    return dataset.select(selected_indices)


def save_running_logs(df_with_errors=None, time_=None, language=None):
    config_instance = Config()

    report_dict = {
        attr: getattr(config_instance, attr)
        for attr in dir(config_instance)
        if not attr.startswith("__") and not callable(getattr(config_instance, attr))
    }

    report_dict["device"] = str(report_dict["device"])

    if df_with_errors:
        errors_list_dict = df_with_errors.to_dict("records")
        report_dict["df_with_errors"] = errors_list_dict
        
        print("\n-------------------error messages-------------------")
        print(df_with_errors, end="\n\n")
        
    if time_:
        report_dict["time"] = time_

        print("-----------------time-----------------")
        print(f"time: {time_:.2f}s")
        
    del report_dict["lang2_dict"]

    if language:
        report_file_path = Config.execution_report_file.replace(".json", f"_{language}.json")
    else:
        report_file_path = Config.execution_report_file
        
    ensure_directory_exists_for_file(report_file_path)
    
    with open(
        report_file_path, "w"
    ) as json_file:
        json.dump(report_dict, json_file, indent=4)

    print(f"Execution report saved to {report_file_path}")
    print(f"Execution report content: {json.dumps(report_dict, indent=4)}")


def main(overwrite, resume_from_checkpoint: bool = None):
    if Config.is_fine_tune:
        print("Starting fine-tune script")
    
        with open(Config.random_seed_path, "r") as f:
            seed = int(f.read())

        print("Loading and preprocessing training dataset")
        train_dataset = load_and_preprocess_dataset(
            Config.finetune_language, split="train"
        )
        print("Loading and preprocessing validation dataset")
        val_matched_dataset = load_and_preprocess_dataset(
            Config.finetune_language, split="validation_matched"
        )

        # few_shot_train_dataset = select_few_shot_samples(
        #     train_dataset["train"], Config.finetune_size, seed
        # )

        data_collator = DataCollatorWithPadding(ModelLoader.get_tokenizer(Config.model))
        
        train_dataloader = DataLoader(train_dataset, batch_size=Config.batch_size, collate_fn=data_collator)
        eval_dataloader = DataLoader(val_matched_dataset, Config.batch_size, collate_fn=data_collator)
        
        
        # checkpoint_dir = Config.finetuned_model_dir
        base_dir = os.path.dirname(Config.finetuned_model_dir)
        checkpoint_dir = get_latest_checkpoint_dir(base_dir)
        checkpoints = [os.path.join(checkpoint_dir, ckpt) for ckpt in os.listdir(checkpoint_dir) if ckpt.startswith("checkpoint-")]
        
        latest_checkpoint = None
        
        if checkpoints:
            use_checkpoint = resume_from_checkpoint or request_user_confirmation(
                f"Fined-tuned model found at {os.path.join(Config.finetuned_model_dir, "model.safetensors")}. Continue from checkpoints? {Color.YELLOW}[Y/n]{Color.END}",
                "Continuing...\n",
                "Skipping checkpoints.\n",
            )
            if use_checkpoint:
                latest_checkpoint = max(checkpoints, key=os.path.getctime)
                print(f"Resuming from checkpoint: {latest_checkpoint}")
                # latest_checkpoint = "fine_tuned_models/MBERT/initial/checkpoint-12000"
            else:
                print("Starting fine-tuning from scratch.")
                
        else:
            print("No checkpoints found. Starting fine-tuning from scratch.")
            
        finetune_model(Config.model, train_dataloader, eval_dataloader, resume_from_checkpoint=latest_checkpoint)
        
        save_running_logs()
        
        return

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
                }
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
        
        results_file_path = Config.results_file.replace(".csv", f"_{language}.csv")
        ensure_directory_exists_for_file(results_file_path)
        df.to_csv(results_file_path, index=False)
        print(
            f"Results stored in {results_file_path}."
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
    os.environ["HF_DATASETS_CACHE"] = "./cache"
    multiprocessing.set_start_method("spawn")
    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    tqdm.monitor_interval = 0
    warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.parallel._functions")

    
    log_filename = os.path.join(Config.logging_dir, f"{'fine-tune' if Config.is_fine_tune else 'evaluate'}_{Config.timestamp}.log")
    logging.basicConfig(
        filename=log_filename,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )
    
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(MultilineFormatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(file_handler)

    # console_handler = logging.StreamHandler()
    # console_handler.setLevel(logging.INFO)
    # console_handler.setFormatter(ColorFormatter('%(asctime)s - %(levelname)s - %(message)s'))
    # logging.getLogger().addHandler(console_handler)
    
    # Clear existing handlers
    logger = logging.getLogger()
    logger.handlers = []
    
    logger.addHandler(file_handler)
    # logger.addHandler(console_handler)
    
    for handler in logging.getLogger().handlers:
        handler.setFormatter(MultilineFormatter('%(asctime)s - %(levelname)s - %(message)s'))
        
    sys.stdout = PrintLogger(sys.stdout, logging.info)
    sys.stderr = PrintLogger(sys.stderr, logging.error)
       
    
    parser = argparse.ArgumentParser(description="Run prediction model.")

    parser.add_argument(
        "-resume_from_checkpoint",
        action="store_true",
        help="Start from checkpoint file if exists.",
    )
    parser.add_argument(
        "-no_resume_from_checkpoint",
        action="store_false",
        dest="resume_from_checkpoint",
        help="Do not start from checkpoint file.",
    )
    parser.set_defaults(resume_from_checkpoint=None)

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

    main(args.overwrite, args.resume_from_checkpoint)
