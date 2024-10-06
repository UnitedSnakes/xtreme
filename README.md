# Reproducing XTREME Benchmark

## Introduction

This repository contains code for reproducing parts of the XTREME (Cross-lingual TRansfer Evaluation of Multilingual Encoders) benchmark. The XTREME benchmark evaluates the performance of multilingual models across a variety of tasks and languages. This repository focuses on tasks like XNLI, Tatoeba, and Multi-NLI using models such as FLAN-UL2 and Multilingual BERT (MBERT).

## Table of Contents

- [Introduction](#introduction)
- [Requirements](#requirements)
- [Project Structure](#project-structure)
- [Datasets](#datasets)
- [Models](#models)
- [Usage](#usage)
- [Results](#results)
- [Acknowledgements](#acknowledgements)

## Requirements

To run the code in this repository, you'll need the following libraries:

- Python 3.8 or later
- PyTorch
- Transformers
- Datasets
- Joblib
- Accelerate
- Sklearn
- Matplotlib
- Pandas

You can install the dependencies using pip:

```bash
pip install -r requirements.txt
```

## Project Structure

- `predict.py`: The main script for running model predictions and evaluations.
- `parameters.py`: Contains configuration parameters used throughout the project.
- `utils.py`: Utility functions for data processing, logging, and other helper functions.
- `README.md`: This document.
- `results/`: Directory to store results and logs.
- `fine_tuned_models/`: Directory to store fine-tuned model checkpoints.

## Datasets

The following datasets are used in this project:

- **XNLI**: Cross-lingual Natural Language Inference.
- **Tatoeba**: Sentence retrieval task.
- **Multi-NLI**: Multi-genre Natural Language Inference.

Datasets can be loaded using the `load_dataset` function from the Hugging Face `datasets` library.

## Models

The following models are supported:

- **FLAN-UL2**: A model for sequence generation tasks.
- **Multilingual BERT (MBERT)**: A BERT model pre-trained on multiple languages.

Models and tokenizers are loaded using the Hugging Face `transformers` library.

## Usage

### Running Predictions

To run predictions using a specified model and dataset, use the `predict.py` script. You can adjust configurations in the `parameters.py` file.

```bash
python predict.py [-resume_from_checkpoint] [-overwrite]
```

- `-resume_from_checkpoint`: Resume training or inference from the last checkpoint if available.
- `-overwrite`: Overwrite existing results.

### Evaluating Predictions

To evaluate the predictions, use the `evaluate.py` script. This script will generate evaluation metrics and store them in the results directory.

```bash
python evaluate.py
```

## Results

The results of the predictions and evaluations are stored in the `results/` directory. This includes:

- Classification reports and confusion matrices for XNLI.
- Accuracy metrics for Tatoeba.
- Logs and execution reports.

## Acknowledgements

This project builds on top of the Hugging Face `transformers` and `datasets` libraries. The XTREME benchmark is an invaluable resource for evaluating the cross-lingual capabilities of multilingual models.

## Contact

For any questions or issues, please contact Shanglin Yang at syang662@wisc.edu.

## scripts


tmux new -s my_session

Ctrl-b d

tmux ls

tmux attach -t my_session
