import torch
import sys
import numpy as np
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from dataset import TokenizedMovieDataset, RandomSubsetDataset
from reshape_movies import get_dictionary_of_movies
from inference import obtain_double_hidden_states_and_labels, load_embedding_model_and_tokenizer
from train_predictor import TokenPredictor

start_movie_id = int(sys.argv[1])
end_movie_id = int(sys.argv[2])
model_name = sys.argv[3]

def calculate_perplexity(logits, labels):
    """
    Calculate perplexity from logits and labels.
    
    Perplexity is a measure of how well a probability model predicts a sample.
    It is defined as exp(cross_entropy_loss).
    
    Args:
        logits (torch.Tensor): Model output logits with shape (N, vocab_size)
        labels (torch.Tensor): True labels with shape (N,)
        
    Returns:
        float: The calculated perplexity
    """
    # Convert logits to probabilities
    probs = torch.softmax(logits, dim=-1)
    
    # Get probability of the actual labels
    label_probs = probs[torch.arange(len(labels)), labels]
    
    # Calculate cross entropy loss
    loss = -torch.mean(torch.log(label_probs))
    
    # Perplexity is exp(loss)
    perplexity = torch.exp(loss)
    
    return perplexity.item()

def calculate_perplexity_numpy(probs, labels):
    """
    Calculate perplexity from probabilities and labels using NumPy.
    
    This function is used for sklearn models that return probabilities directly.
    
    Args:
        probs (numpy.ndarray): Model output probabilities with shape (N, vocab_size)
        labels (numpy.ndarray): Labels with shape (N,)
        
    Returns:
        float: The calculated perplexity
    """
    # Get probability of the actual labels
    label_probs = probs[np.arange(len(labels)), labels]
    
    # Calculate cross entropy loss
    loss = -np.mean(np.log(label_probs + 1e-10))  # Add small epsilon to avoid log(0)
    
    # Perplexity is exp(loss)
    perplexity = np.exp(loss)
    
    return perplexity

def get_hidden_states_and_labels(model, tokenizer, dataset, batch_size=4):
    """
    Get hidden states and labels from the dataset.
    """
    hidden_states, labels = obtain_double_hidden_states_and_labels(model, tokenizer, dataset, batch_size=batch_size)
    hidden_states = torch.tensor(hidden_states, dtype=torch.float16)
    labels = torch.tensor(labels, dtype=torch.float16)
    return hidden_states, labels

def evaluate_perplexity(model, tokenizer, predictor, hidden_states, labels, batch_size=4):
    """
    Evaluate perplexity on the dataset.
    
    This function:
    1. Obtains hidden states and labels from the dataset
    2. Uses the predictor to generate logits
    3. Calculates perplexity from the logits and labels
    
    Args:
        model: The embedding model
        tokenizer: The tokenizer for encoding/decoding text
        predictor: The trained token predictor
        dataset: The dataset to evaluate on
        batch_size (int, optional): Batch size for processing. Defaults to 4.
        
    Returns:
        float: The calculated perplexity
    """
    
    print("Computing logits and calculating perplexity...")
    with torch.no_grad():
        logits = predictor(hidden_states)
        perplexity = calculate_perplexity(logits, labels)
    
    return perplexity

def evaluate_lr_perplexity(lr_model, hidden_states, labels):
    """
    Evaluate perplexity on the dataset using a logistic regression model.
    
    This function:
    1. Uses the logistic regression model to predict probabilities
    2. Calculates perplexity from the probabilities and labels
    
    Args:
        lr_model: The trained logistic regression model
        hidden_states (torch.Tensor): Hidden states with shape (N, D)
        labels (torch.Tensor): Labels with shape (N,)
        
    Returns:
        float: The calculated perplexity
    """
    print("Computing probabilities and calculating perplexity...")
    
    # Convert to numpy for sklearn
    hidden_states_np = hidden_states.cpu().numpy()
    labels_np = labels.cpu().numpy()
    
    # Get the number of classes in the logistic regression model
    n_classes = lr_model.classes_.shape[0]
    print(f"Logistic regression model has {n_classes} classes")
    
    # Filter out labels that are outside the model's vocabulary range
    valid_indices = labels_np < n_classes
    if not np.all(valid_indices):
        print(f"Warning: {np.sum(~valid_indices)} labels are outside the model's vocabulary range")
        print(f"These labels will be excluded from perplexity calculation")
        hidden_states_np = hidden_states_np[valid_indices]
        labels_np = labels_np[valid_indices]
        
        if len(labels_np) == 0:
            print("Error: No valid labels remain after filtering")
            return float('inf')
    
    # Get probabilities from the model
    probs = lr_model.predict_proba(hidden_states_np)
    
    # Calculate perplexity
    perplexity = calculate_perplexity_numpy(probs, labels_np)
    
    return perplexity

def evaluate_gpt2_perplexity(model, tokenizer, dataset, batch_size=1):
    """
    Evaluate perplexity on the dataset using a GPT-2 model.
    
    This function:
    1. Processes the dataset to get input sequences and labels
    2. Passes the input sequences through the GPT-2 model
    3. Extracts the logits from the model output
    4. Calculates perplexity from the logits and labels
    
    Args:
        model: The GPT-2 model
        tokenizer: The tokenizer for encoding/decoding text
        dataset: The dataset to evaluate on
        batch_size (int, optional): Batch size for processing. Defaults to 1.
        
    Returns:
        float: The calculated perplexity
    """
    from torch.utils.data import DataLoader
    
    # Create a random subset of the dataset for evaluation
    # This helps reduce memory usage and computation time
    subset_size = min(100, len(dataset))  # Use at most 100 items or the full dataset if smaller
    subset_dataset = RandomSubsetDataset(dataset, size=subset_size, seed=42)  # Fixed seed for reproducibility
    
    # Create a dataloader for the dataset
    dataloader = DataLoader(subset_dataset, batch_size=batch_size, shuffle=False)
    
    # Move model to the appropriate device
    device = next(model.parameters()).device
    model.eval()
    
    # Enable gradient checkpointing to save memory
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    
    all_logits = []
    all_labels = []
    
    print("Computing logits and calculating perplexity...")
    with torch.no_grad():
        for batch in dataloader:
            # Get input IDs and attention mask
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            #print("Input ids: ",input_ids.shape)
            
            # Get model outputs
            outputs = model(input_ids, attention_mask=attention_mask)
            
            # Extract logits from the output
            # For GPT-2, logits are in outputs.logits
            # We need to get the logits for the next token prediction
            # Shape: (batch_size, seq_len, vocab_size)
            logits = outputs.logits
            
            # Get the logits for the position of each label
            # We need to shift the input_ids to get the correct positions
            # For each sequence, we want the logits at position i-1 to predict token at position i
            shifted_logits = logits[:, :-1, :]
            shifted_labels = input_ids[:, 1:]
            
            # Store the logits and labels
            all_logits.append(shifted_logits.reshape(-1, shifted_logits.size(-1)))
            all_labels.append(shifted_labels.reshape(-1))
    
    # Concatenate all logits and labels
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # Calculate perplexity
    perplexity = calculate_perplexity(all_logits, all_labels)
    
    return perplexity

def main():
    """
    Main function to evaluate perplexity on held-out data.
    
    This function:
    1. Loads the embedding model and tokenizer
    2. Loads the trained predictor
    3. Loads a range of movies for evaluation
    4. Calculates and prints the perplexity
    
    Command line arguments:
        sys.argv[1]: start_movie_id - The starting movie ID
        sys.argv[2]: end_movie_id - The ending movie ID
        sys.argv[3]: model_name - The name of the model used
    """
    # Load models
    print("Loading models...")
    model, tokenizer = load_embedding_model_and_tokenizer(model_name)
    
    # Load evaluation dataset
    print("Loading evaluation dataset...")
    all_movies = get_dictionary_of_movies(N_MOVIES_TO_USE=end_movie_id - start_movie_id, start_movie_id=start_movie_id)
    dataset = TokenizedMovieDataset(all_movies, tokenizer, padding=True, max_length=300, min_length=200)
    del all_movies
    # Take a random subset of 10 samples from the dataset
    indices = torch.randperm(len(dataset))[:10]
    dataset = torch.utils.data.Subset(dataset, indices)
    
    # Load predictor (make sure to load state dict correctly)
    print("Loading predictor...")
    predictor = TokenPredictor(input_dim=1536, vocab_size=tokenizer.vocab_size).to('cuda')  # 1536 = 2 * 768 (GPT-2 hidden size).t
    predictor.load_state_dict(torch.load(f"../token_predictor_{model_name}_gd.pt"))
    predictor.eval()
    
    # Load logistic regression model
    print("Loading logistic regression model...")
    lr_model = np.load(f"../token_predictor_{model_name}_lr.npy", allow_pickle=True).item()
    
    ## Get hidden states and labels
    print("Getting hidden states and labels...")
    hidden_states, labels = get_hidden_states_and_labels(model, tokenizer, dataset)
    hidden_states = hidden_states.to('cuda')
    labels = labels.to('cuda')
    
    # Evaluate perplexity of GD model
    print("Evaluating perplexity...")
    perplexity = evaluate_perplexity(model, tokenizer, predictor, hidden_states, labels)
    
    print(f"Perplexity on held-out data, using token predictor (GD): {perplexity:.4f}")
    
    # Evaluate perplexity of LR model
    print("Evaluating perplexity of LR model...")
    lr_perplexity = evaluate_lr_perplexity(lr_model, hidden_states, labels)
    
    print(f"Perplexity on held-out data, using token predictor (LR): {lr_perplexity:.4f}")
    
    # Evaluate perplexity using basic model
    print("Loading GPT-2 model for direct perplexity evaluation...")
    gpt2_model = AutoModelForCausalLM.from_pretrained(model_name).to('cuda')
    gpt2_tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Set padding token if not set
    if gpt2_tokenizer.pad_token is None:
        gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
        gpt2_model.config.pad_token_id = gpt2_model.config.eos_token_id
    
    gpt2_perplexity = evaluate_gpt2_perplexity(gpt2_model, gpt2_tokenizer, dataset)
    
    print(f"Perplexity on held-out data, using basic model: {gpt2_perplexity:.4f}")

if __name__ == "__main__":
    main()