import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from torch.utils.data import DataLoader
from dataset import TokenizedMovieDataset
from reshape_movies import get_dictionary_of_movies
import numpy as np

def load_model_and_tokenizer(model_name="ai21labs/Jamba-v0.1"):
    """Load the model and tokenizer from Hugging Face."""
    print(f"Loading model and tokenizer from {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True  # Needed for JAMBA architecture
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Set up padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    
    return model, tokenizer

def load_embedding_model_and_tokenizer(model_name="ai21labs/Jamba-v0.1"):
    """Load the embedding model and tokenizer from Hugging Face."""
    model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True  # Needed for JAMBA architecture
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Set up padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    
    return model, tokenizer
def perform_inference(model, tokenizer, dataset, batch_size=4, max_length=512):
    """Perform inference on the dataset using the model, jamba or other"""
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
    """Obtain the hidden states of the model."""
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

def obtain_double_hidden_states(model, tokenizer, dataset, batch_size=4, max_length=512):
    """Obtain the double hidden states of the model."""
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    
    all_hidden_states = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device=model.device)
            attention_mask_1 = obtain_all_but_last_attention_mask(input_ids, batch['token_speaker_ids']).to(device=model.device)
            attention_mask_2 = obtain_character_specific_attention_mask(input_ids, batch['token_speaker_ids']).to(device=model.device)
            hidden_states_1 = model(input_ids, attention_mask=attention_mask_1).last_hidden_state
            hidden_states_2 = model(input_ids, attention_mask=attention_mask_2).last_hidden_state
            all_hidden_states.append(np.concatenate([hidden_states_1, hidden_states_2], axis=1))
    
    return torch.cat(all_hidden_states, dim=0)
def obtain_all_but_last_attention_mask(input_ids, speaker_ids):
    mask = torch.ones_like(input_ids)
    mask[:,-1] = 0
    return mask

def obtain_character_specific_attention_mask(input_ids, speaker_ids):
    """Obtain the character specific attention mask of the model."""
    last_speaker_ids = speaker_ids[:,-1].unsqueeze(1)
    char_specific_mask = speaker_ids == last_speaker_ids
    char_specific_mask = char_specific_mask.unsqueeze(1)
    return char_specific_mask

def main():
    # Load model and tokenizer, doesn't have to be jamba
    model, tokenizer = load_embedding_model_and_tokenizer("gpt2")
    
    # Get the movie dataset
    print("Loading movie dataset...")
    all_movies = get_dictionary_of_movies(N_MOVIES_TO_USE=10)  # Start with 10 movies for testing
    dataset = TokenizedMovieDataset(all_movies, tokenizer, padding=True, max_length=512)
    '''
    # Perform inference
    print("Performing inference...")
    outputs = perform_inference(model, tokenizer, dataset)
    
    # Save results
    print("Saving results...")
    with open("inference_results.txt", "w") as f:
        for i, output in enumerate(outputs):
            f.write(f"Movie {i+1}:\n{output}\n\n")
    '''
    # Obtain hidden states
    print("Obtaining hidden states...")
    hidden_states = obtain_double_hidden_states(model, tokenizer, dataset)
    
    # Save hidden states
    print("Saving hidden states...")
    np.save("double_hidden_states.npy", hidden_states.cpu())

if __name__ == "__main__":
    main() 