import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from torch.utils.data import DataLoader
from dataset import TokenizedMovieDataset, RandomSubsetDataset
from reshape_movies import get_dictionary_of_movies
import numpy as np
import os, sys

def load_model_and_tokenizer(model_name="ai21labs/Jamba-v0.1"):
    """
    Load a causal language model and tokenizer from Hugging Face.
    
    This function loads a model suitable for text generation and its corresponding tokenizer.
    It sets up the padding token if needed.
    
    Args:
        model_name (str, optional): Name of the model to load. Defaults to "ai21labs/Jamba-v0.1".
        
    Returns:
        tuple: (model, tokenizer) - The loaded model and tokenizer
    """
    print(f"Loading model and tokenizer from {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True  # Needed for JAMBA architecture
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Set up padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    
    return model, tokenizer

def load_embedding_model_and_tokenizer(model_name="ai21labs/Jamba-v0.1"):
    """
    Load an embedding model and tokenizer from Hugging Face.
    
    This function loads a model suitable for extracting embeddings and its corresponding tokenizer.
    It disables Flash Attention for compatibility with non-Ampere GPUs.
    
    Args:
        model_name (str, optional): Name of the model to load. Defaults to "ai21labs/Jamba-v0.1".
        
    Returns:
        tuple: (model, tokenizer) - The loaded model and tokenizer
    """
    # Disable Flash Attention
    os.environ["TRANSFORMERS_NO_FLASH_ATTN"] = "1"
    
    model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True  # Needed for JAMBA architecture
    ).to('cuda')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Set up padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    
    return model, tokenizer

def perform_inference(model, tokenizer, dataset, batch_size=4, max_length=512):
    """
    Perform inference on the dataset using the model.
    
    This function generates text for each item in the dataset using the model.
    
    Args:
        model: The language model to use for generation
        tokenizer: The tokenizer for encoding/decoding text
        dataset: The dataset to perform inference on
        batch_size (int, optional): Batch size for processing. Defaults to 4.
        max_length (int, optional): Maximum sequence length. Defaults to 512.
        
    Returns:
        list: List of generated text sequences
    """
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    
    all_outputs = []
    
    with torch.no_grad():
        for batch in dataloader:
            # Get input IDs and create attention masks
            input_ids = batch['input_ids']  # Already a tensor from dataset
            attention_mask = torch.ones_like(input_ids)
            
            # Move to the same device as the model
            input_ids = input_ids.to(model.device)
            attention_mask = attention_mask.to(model.device)
            
            # Generate outputs
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_length=max_length+100,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
            
            # Decode the outputs
            decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            all_outputs.extend(decoded_outputs)
            
    return all_outputs

def obtain_hidden_states(model, tokenizer, dataset, batch_size=4, max_length=512):
    """
    Obtain the hidden states of the model for each item in the dataset.
    
    This function extracts the hidden states from the model for each item in the dataset.
    
    Args:
        model: The model to extract hidden states from
        tokenizer: The tokenizer for encoding/decoding text
        dataset: The dataset to process
        batch_size (int, optional): Batch size for processing. Defaults to 4.
        max_length (int, optional): Maximum sequence length. Defaults to 512.
        
    Returns:
        torch.Tensor: Tensor of hidden states with shape (N, L, D) where N is the number of items,
                     L is the sequence length, and D is the hidden dimension
    """
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    
    all_hidden_states = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids']
            attention_mask = torch.ones_like(input_ids)
            attention_mask[:,-1] = 0
            input_ids = input_ids.to(model.device)
            attention_mask = attention_mask.to(model.device)
            hidden_states = model(input_ids, attention_mask=attention_mask).last_hidden_state
            all_hidden_states.append(hidden_states)
    
    return torch.cat(all_hidden_states, dim=0)

def obtain_double_hidden_states_and_labels(model, tokenizer, dataset, batch_size=4):
    """
    Obtain both speaker-specific and general hidden states, along with labels.
    
    This function extracts two sets of hidden states for each item in the dataset:
    1. Speaker-specific hidden states using the speaker mask
    2. General hidden states using the attention mask
    
    It also returns the labels for each item.
    
    Args:
        model: The model to extract hidden states from
        tokenizer: The tokenizer for encoding/decoding text
        dataset: The dataset to process
        batch_size (int, optional): Batch size for processing. Defaults to 4.
        
    Returns:
        tuple: (hidden_states, labels) - numpy arrays of hidden states and labels
               hidden_states has shape (N, 2D) where N is the number of items and D is the hidden dimension
               labels has shape (N,) where N is the number of items
    """
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    
    all_hidden_states = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device=model.device)
            labels = batch['label'].to(device=model.device)
            speaker_mask = batch['speaker_mask'].to(device=model.device)
            attention_mask = batch['attention_mask'].to(device=model.device)
            
            # Use attention mask for general hidden states
            hidden_states_speaker = model(input_ids, attention_mask=speaker_mask).last_hidden_state  # (B,L,D)
            hidden_states_general = model(input_ids, attention_mask=attention_mask).last_hidden_state  # (B,L,D)
            
            # Concatenate along last dimension to get (B,L,2D)
            hidden_states_concat = torch.cat([hidden_states_speaker, hidden_states_general], dim=-1)
            
            # Mean pool across sequence length dimension to get (B,2D)
            hidden_states_concat = hidden_states_concat.mean(dim=1)
            
            all_hidden_states.append(hidden_states_concat.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    return np.concatenate(all_hidden_states, axis=0), np.concatenate(all_labels, axis=0)

def main():
    """
    Main function to process a range of movies, extract hidden states, and save them.
    
    This function:
    1. Loads a model and tokenizer
    2. Loads a range of movies from the dataset
    3. Extracts hidden states and labels
    4. Saves the results to files
    
    Command line arguments:
        sys.argv[1]: start_movie_id - The starting movie ID
        sys.argv[2]: end_movie_id - The ending movie ID
        sys.argv[3]: model_name - The name of the model to use
    """
    
    start_movie_id = int(sys.argv[1])  
    end_movie_id = int(sys.argv[2])
    model_name = sys.argv[3]

    # Load model and tokenizer, doesn't have to be jamba
    model, tokenizer = load_embedding_model_and_tokenizer(model_name)
    
    # Get the movie dataset
    print("Loading movie dataset...")
    all_movies = get_dictionary_of_movies(N_MOVIES_TO_USE=end_movie_id - start_movie_id, start_movie_id=start_movie_id)  # Start with 10 movies for testing
    dataset = TokenizedMovieDataset(all_movies, tokenizer, padding=True, max_length=300, min_length=200)
    
    dataset = RandomSubsetDataset(dataset, size=10000)
    
    # Obtain hidden states
    print("Obtaining hidden states...")
    hidden_states, labels = obtain_double_hidden_states_and_labels(model, tokenizer, dataset)
    
    # Save hidden states
    print("Saving hidden states...")
    np.save(f"double_hidden_states_{model_name}.npy", hidden_states)
    np.save(f"labels_{model_name}.npy", labels)

if __name__ == "__main__":
    main()