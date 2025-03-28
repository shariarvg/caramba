import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader
from dataset import TokenizedMovieDataset
from reshape_movies import get_dictionary_of_movies
import numpy as np

def load_model_and_tokenizer(model_name="ai21labs/Jamba-v0.1"):
    """Load the JAMBA model and tokenizer from Hugging Face."""
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

def main():
    # Load model and tokenizer, doesn't have to be jamba
    model, tokenizer = load_model_and_tokenizer("gpt2")
    
    # Get the movie dataset
    print("Loading movie dataset...")
    all_movies = get_dictionary_of_movies(N_MOVIES_TO_USE=10)  # Start with 10 movies for testing
    dataset = TokenizedMovieDataset(all_movies, tokenizer, padding=True, max_length=512)
    
    # Perform inference
    print("Performing inference...")
    outputs = perform_inference(model, tokenizer, dataset)
    
    # Save results
    print("Saving results...")
    with open("inference_results.txt", "w") as f:
        for i, output in enumerate(outputs):
            f.write(f"Movie {i+1}:\n{output}\n\n")

if __name__ == "__main__":
    main() 