import torch
import torch.nn.functional as F
import time
import numpy as np
from transformers import LongformerForMaskedLM, LongformerTokenizerFast, LongformerConfig
from dataset import TokenizedMovieDatasetForMaskedLM, RandomSubsetDataset
from reshape_movies import get_dictionary_of_movies
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm

def load_longformer_model(model_name="allenai/longformer-base-4096"):
    """
    Load the Longformer model and tokenizer.
    
    Args:
        model_name (str): Name of the Longformer model to load
        
    Returns:
        tuple: (model, tokenizer)
    """
    print(f"Loading Longformer model: {model_name}")
    # Load the model and tokenizer
    

    # Load config and set a custom local attention window (e.g., 5 tokens on each side)
    config = LongformerConfig.from_pretrained("allenai/longformer-base-4096")
    config.attention_window = [100] * config.num_hidden_layers  # 5 = window size per layer

    # Load model with modified config
    model = LongformerForMaskedLM.from_pretrained("allenai/longformer-base-4096", config=config)

    tokenizer = LongformerTokenizerFast.from_pretrained(model_name)
    
    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    
    return model, tokenizer

def calculate_cross_entropy_loss(logits, labels):
    """
    Calculate cross entropy loss between logits and labels.
    
    Args:
        logits (torch.Tensor): Model output logits
        labels (torch.Tensor): Target labels
        
    Returns:
        float: Cross entropy loss
    """
    # Reshape logits and labels for cross entropy
    logits_flat = logits.view(-1, logits.size(-1))
    labels_flat = labels.view(-1)
    
    # Calculate loss
    loss = F.cross_entropy(logits_flat, labels_flat)
    
    return loss.item()

def predict_next_tokens_full_attention(model, tokenizer, dataset, batch_size=4):
    """
    Predict next tokens using Longformer with full global attention.
    
    Args:
        model: Longformer model
        tokenizer: Longformer tokenizer
        dataset: Dataset containing tokenized inputs
        batch_size (int): Batch size for processing
        
    Returns:
        tuple: (loss, runtime)
    """
    # Create a dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    total_loss = 0
    total_tokens = 0
    start_time = time.time()
    
    # Process each batch
    for batch in tqdm(dataloader, desc="Full Attention"):
        # Move inputs to device
        input_ids = batch["input_ids"].to(device)
        labels = batch["label"].to(device)
        
        # Ensure input_ids has shape [batch_size, seq_length]
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)  # Add batch dimension
        
        # Ensure labels has shape [batch_size]
        if labels.dim() == 0:
            labels = labels.unsqueeze(0)  # Add batch dimension
        
        # Create attention mask (1 for actual tokens, 0 for padding)
        attention_mask = (input_ids != tokenizer.pad_token_id).long()
        
        # Set global attention mask to 1 for all tokens (full attention)
        global_attention_mask = torch.ones_like(input_ids)
        
        
        # Forward pass
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                global_attention_mask=global_attention_mask,
                labels=labels
            )
        
        # Get loss
        loss = outputs.loss.item()
        total_loss += loss #* input_ids.size(0)
        #total_tokens += input_ids.size(0)
    
    # Calculate runtime
    runtime = time.time() - start_time
    
    # Calculate average loss
    #avg_loss = total_loss / total_tokens
    
    return total_loss / len(dataset) * batch_size, runtime

def predict_next_tokens_hybrid_attention(model, tokenizer, dataset, batch_size=4):
    """
    Predict next tokens using Longformer with hybrid attention:
    - Local attention to the last local_window tokens
    - Global attention to all tokens spoken by the same character
    
    Args:
        model: Longformer model
        tokenizer: Longformer tokenizer
        dataset: Dataset containing tokenized inputs
        batch_size (int): Batch size for processing
        
    Returns:
        tuple: (loss, runtime)
    """
    # Create a dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    total_loss = 0
    total_tokens = 0
    start_time = time.time()
    
    # Process each batch
    for batch in tqdm(dataloader, desc="Hybrid Attention"):
        # Move inputs to device
        input_ids = batch["input_ids"].to(device)
        labels = batch["label"].to(device)
        speaker_mask = batch["speaker_mask"].to(device)
        
        # Ensure input_ids has shape [batch_size, seq_length]
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)  # Add batch dimension
        
        # Ensure labels has shape [batch_size]
        if labels.dim() == 0:
            labels = labels.unsqueeze(0)  # Add batch dimension
        
        # Ensure speaker_mask has shape [batch_size, seq_length]
        if speaker_mask.dim() == 1:
            speaker_mask = speaker_mask.unsqueeze(0)  # Add batch dimension
        
        attention_mask = (input_ids != tokenizer.pad_token_id).long()

        # Create global attention mask
        # 1 for tokens with global attention, 0 for tokens with local attention
        global_attention_mask = speaker_mask
        
        # Forward pass
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                global_attention_mask=global_attention_mask,
                labels=labels
            )
        
        # Get loss
        loss = outputs.loss.item()
        total_loss += loss #* input_ids.size(0)
        #total_tokens += input_ids.size(0)
    
    # Calculate runtime
    runtime = time.time() - start_time
    
    # Calculate average loss
    #avg_loss = total_loss / total_tokens
    
    return total_loss / len(dataset) * batch_size, runtime

def predict_next_tokens_random_hybrid_attention(model, tokenizer, dataset, random_hybrid_ratio=0.8, batch_size=4):
    """
    Predict next tokens using Longformer with random hybrid attention:
    - Local attention to the last local_window tokens
    - Global attention to a random subset of tokens (0.8 probability)
    
    """
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    total_loss = 0
    total_tokens = 0
    start_time = time.time()
    
    # Process each batch
    
    for batch in tqdm(dataloader, desc="Hybrid Attention"):
        # Move inputs to device
        input_ids = batch["input_ids"].to(device)
        labels = batch["label"].to(device)
        speaker_mask = batch["speaker_mask"].to(device)
        
        # Ensure input_ids has shape [batch_size, seq_length]
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)  # Add batch dimension
        
        # Ensure labels has shape [batch_size]
        if labels.dim() == 0:
            labels = labels.unsqueeze(0)  # Add batch dimension
        
        attention_mask = (input_ids != tokenizer.pad_token_id).long()

        # Create global attention mask
        # 1 for tokens with global attention, 0 for tokens with local attention
        global_attention_mask = (torch.rand(input_ids.shape, device=input_ids.device) < random_hybrid_ratio).int()
        global_attention_mask[:,-1]=1 
        
        # Forward pass
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                global_attention_mask=global_attention_mask,
                labels=labels
            )
        
        # Get loss
        loss = outputs.loss.item()
        total_loss += loss #* input_ids.size(0)
        #total_tokens += input_ids.size(0)
    
    # Calculate runtime
    runtime = time.time() - start_time
    
    # Calculate average loss
    #avg_loss = total_loss / total_tokens
    
    return total_loss / len(dataset) * batch_size, runtime
def obtain_predictions(model, tokenizer, dataset, max_length=300, local_window=100, batch_size=1):
    """
    Obtain predictions using both full attention and hybrid attention.
    
    Args:
        model: Longformer model
        tokenizer: Longformer tokenizer
        dataset: Dataset containing tokenized inputs
        max_length (int): Maximum sequence length
        local_window (int): Number of tokens for local attention
        batch_size (int): Batch size for processing
        
    Returns:
        tuple: (full_attention_results, hybrid_attention_results)
    """
    
    # Process with full attention
    full_loss, full_runtime = predict_next_tokens_full_attention(
        model, tokenizer, dataset, batch_size
    )
    
    # Process with hybrid attention
    hybrid_loss, hybrid_runtime = predict_next_tokens_hybrid_attention(
        model, tokenizer, dataset, batch_size
    )
    
    # Process with random hybrid attention
    random_hybrid_loss, random_hybrid_runtime = predict_next_tokens_random_hybrid_attention(
        model, tokenizer, dataset, batch_size
    )
    
    return (full_loss, full_runtime), (hybrid_loss, hybrid_runtime), (random_hybrid_loss, random_hybrid_runtime)

def main():
    parser = argparse.ArgumentParser(description="Compare Longformer attention strategies")
    parser.add_argument("--model", type=str, default="allenai/longformer-base-4096", help="Longformer model to use")
    parser.add_argument("--max_length", type=int, default=500, help="Maximum sequence length")
    parser.add_argument("--local_window", type=int, default=100, help="Number of tokens for local attention")
    parser.add_argument("--num_movies", type=int, default=10, help="Number of movies to process")
    parser.add_argument("--start_movie_id", type=int, default=0, help="Starting movie ID")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for processing")
    args = parser.parse_args()
    
    # Load model and tokenizer
    model, tokenizer = load_longformer_model(args.model)
    
    # Load movie dataset
    print(f"Loading {args.num_movies} movies starting from ID {args.start_movie_id}...")
    all_movies = get_dictionary_of_movies(
        N_MOVIES_TO_USE=args.num_movies,
        start_movie_id=args.start_movie_id
    )
    dataset = TokenizedMovieDatasetForMaskedLM(all_movies, tokenizer, padding=True, max_length=args.max_length, min_length=100)
    dataset = RandomSubsetDataset(dataset, size=1000)
    # Process each movie
    
    (full_loss, full_runtime), (hybrid_loss, hybrid_runtime), (random_hybrid_loss, random_hybrid_runtime) = obtain_predictions(
        model, tokenizer, dataset, args.max_length, args.local_window, args.batch_size
    )
    
    
    print("\nResults Summary:")
    print(f"Full Attention - Average Loss: {full_loss:.4f}, Average Runtime: {full_runtime:.4f}s")
    print(f"Hybrid Attention - Average Loss: {hybrid_loss:.4f}, Average Runtime: {hybrid_runtime:.4f}s")
    print(f"Runtime Improvement: {(full_runtime - hybrid_runtime) / full_runtime * 100:.2f}%")
    print(f"Loss Difference: {hybrid_loss - full_loss:.4f}")
    print(f"Random Hybrid Attention - Average Loss: {random_hybrid_loss:.4f}, Average Runtime: {random_hybrid_runtime:.4f}s")
    print(f"Runtime Improvement: {(full_runtime - random_hybrid_runtime) / full_runtime * 100:.2f}%")
    print(f"Loss Difference: {random_hybrid_loss - full_loss:.4f}")
    
    # Plot results
    plt.figure(figsize=(12, 5))
    
    # Plot losses
    plt.subplot(1, 2, 1)
    plt.bar(["Full Attention", "Hybrid Attention", "Random Hybrid Attention"], [full_loss, hybrid_loss, random_hybrid_loss])
    plt.title("Average Cross-Entropy Loss")
    plt.ylabel("Loss")
    
    # Plot runtimes
    plt.subplot(1, 2, 2)
    plt.bar(["Full Attention", "Hybrid Attention", "Random Hybrid Attention"], [full_runtime, hybrid_runtime, random_hybrid_runtime])
    plt.title("Average Runtime")
    plt.ylabel("Seconds")
    
    plt.tight_layout()
    plt.savefig("longformer_comparison.png")
    print("Results plot saved as 'longformer_comparison.png'")

if __name__ == "__main__":
    main() 