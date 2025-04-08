# Movie Dialogue Analysis and Perplexity Evaluation

This repository contains code for analyzing movie dialogues, extracting hidden states from language models, and evaluating perplexity on movie dialogue datasets.

## Project Overview

This project focuses on:
1. Processing movie dialogue data
2. Extracting hidden states from language models
3. Training token predictors
4. Evaluating perplexity on movie dialogue datasets

## Repository Structure

- `dataset.py`: Contains the `TokenizedMovieDataset` class for processing movie dialogue data
- `inference.py`: Functions for loading models and extracting hidden states
- `train_predictor.py`: Code for training token predictors
- `evaluate_perplexity.py`: Functions for evaluating perplexity on movie dialogue datasets
- `reshape_movies.py`: Utilities for processing movie dialogue data

## Requirements

- Python 3.6+
- PyTorch
- Transformers
- NumPy
- Convokit (for movie dialogue data)
- CUDA-compatible GPU (recommended)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-directory>
```

2. Install dependencies:
```bash
pip install torch transformers numpy convokit
```

## Usage

### Data Processing

The project uses the Convokit movie corpus for dialogue data. The data is processed using the `TokenizedMovieDataset` class in `dataset.py`.

### Extracting Hidden States

To extract hidden states from a language model:

```bash
python inference.py <start_movie_id> <end_movie_id> <model_name>
```

This will:
1. Load the specified model
2. Process the movie dialogue data
3. Extract hidden states and labels
4. Save the results to files

### Training Token Predictors

To train token predictors:

```bash
python train_predictor.py <model_name>
```

This will:
1. Load the hidden states and labels
2. Train a neural network using gradient descent
3. Train a logistic regression model
4. Save both models

### Evaluating Perplexity

To evaluate perplexity on movie dialogue datasets:

```bash
python evaluate_perplexity.py <start_movie_id> <end_movie_id> <model_name>
```

This will:
1. Load the specified model
2. Process the movie dialogue data
3. Calculate perplexity using the model
4. Print the results

## Memory Optimization

The code includes memory optimization techniques for handling large models and datasets:

- Gradient checkpointing to reduce memory usage
- Chunked processing for long sequences
- Periodic CUDA cache clearing
- Reduced batch sizes for large models

## Scripts

- `inference.sh`: Script for running the inference pipeline
- `train_predictor.sh`: Script for training token predictors
- `evaluate_predictor.sh`: Script for evaluating token predictors

## Notes

- The code is optimized for CUDA-compatible GPUs
- Memory usage can be high when processing large models or datasets
- Adjust batch sizes and chunk sizes based on available GPU memory
