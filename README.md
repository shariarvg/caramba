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

## Dataset Classes

### TokenizedMovieDataset

The base `TokenizedMovieDataset` class processes movie dialogue data and prepares it for token prediction tasks. It:

1. Tokenizes movie dialogue text using a specified tokenizer
2. Maps speaker IDs to sequential integers
3. Creates data points for each position in the dialogue
4. Returns:
   - `input_ids`: Tokens up to the current position
   - `label`: The next token to predict
   - `speaker_mask`: Binary mask indicating which previous tokens share the same speaker
   - `attention_mask`: Binary mask indicating actual tokens vs padding

### TokenizedMovieDatasetForMaskedLM

`TokenizedMovieDatasetForMaskedLM` extends `TokenizedMovieDataset` to support masked language modeling. Key differences:

1. Adds a mask token at the end of each sequence
2. Modifies the label format:
   - Original: Single token ID to predict
   - MaskedLM: Full sequence with mask token and -100 for non-masked positions
3. Adjusts speaker masks to include the mask token position
4. Designed specifically for models like Longformer that use masked language modeling

### Key Differences

| Feature | TokenizedMovieDataset | TokenizedMovieDatasetForMaskedLM |
|---------|----------------------|----------------------------------|
| Output Format | Single token prediction | Masked language modeling |
| Label Shape | Scalar (next token) | Sequence with mask token |
| Speaker Mask | Original sequence | Extended for mask token |
| Use Case | Next token prediction | Masked language modeling |

## Example: Evaluating an Autoregressive Transformer

Here's an example of how to use `TokenizedMovieDataset` to evaluate an off-the-shelf autoregressive transformer model (like GPT-2) on movie dialogue:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataset import TokenizedMovieDataset
from reshape_movies import get_dictionary_of_movies
import torch

# Load model and tokenizer
model_name = "gpt2"  # or any other autoregressive model
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load movie data
all_movies = get_dictionary_of_movies(
    N_MOVIES_TO_USE=10,  # Number of movies to process
    start_movie_id=0     # Starting movie ID
)

# Create dataset
dataset = TokenizedMovieDataset(
    all_movies=all_movies,
    tokenizer=tokenizer,
    padding=True,        # Pad sequences to max_length
    max_length=512,      # Maximum sequence length
    min_length=100       # Minimum sequence length
)

# Create dataloader
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=4,        # Adjust based on GPU memory
    shuffle=False
)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Evaluate model
total_loss = 0
total_tokens = 0

for batch in dataloader:
    # Move inputs to device
    input_ids = batch["input_ids"].to(device)
    labels = batch["label"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    
    # Forward pass
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
    
    # Accumulate loss
    total_loss += outputs.loss.item() * input_ids.size(0)
    total_tokens += input_ids.size(0)

# Calculate average loss
avg_loss = total_loss / total_tokens
print(f"Average loss: {avg_loss:.4f}")
```

This example:
1. Loads a pre-trained autoregressive model and its tokenizer
2. Creates a `TokenizedMovieDataset` from movie dialogue data
3. Sets up a dataloader for batch processing
4. Evaluates the model on the dataset, calculating the average loss

The dataset automatically handles:
- Tokenization of movie dialogue
- Creation of input sequences and labels
- Speaker tracking and masking
- Sequence padding and truncation

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
