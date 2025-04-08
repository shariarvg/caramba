import torch
from transformers import AutoTokenizer
from dataset import TokenizedMovieDataset
from reshape_movies import get_dictionary_of_movies
import numpy as np
import os

def create_small_dataset():
    """Create and save a small subset of movies for testing."""
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    
    print("Loading movie data...")
    # Get a small subset of movies
    small_movies = get_dictionary_of_movies(
        N_MOVIES_TO_USE=20,  # Just 20 movies
        N_CONVOS=1000,       # Limit total conversations
        start_movie_id=0     # Start from the beginning
    )
    
    # Create tokenized dataset
    print("Creating tokenized dataset...")
    tokenized_dataset = TokenizedMovieDataset(
        small_movies,
        tokenizer,
        padding=True,
        max_length=50  # Reduced max length for testing
    )
    
    # Save the raw movie data
    print("Saving raw movie data...")
    torch.save(small_movies, "small_movies.pt")
    
    # Save the tokenized dataset
    print("Saving tokenized dataset...")
    torch.save(tokenized_dataset, "small_tokenized_dataset.pt")
    
    # Print some statistics
    print("\nDataset Statistics:")
    print(f"Number of movies: {len(small_movies['text'])}")
    print(f"Total conversations: {sum(len(movie) for movie in small_movies['text'])}")
    print(f"Average conversations per movie: {np.mean([len(movie) for movie in small_movies['text']]):.2f}")
    
    # Print a sample from the first movie
    print("\nSample from first movie:")
    print("-" * 50)
    for i in range(min(5, len(small_movies['text'][0]))):
        print(f"{small_movies['speaker_ids'][0][i]}: {small_movies['text'][0][i]}")

if __name__ == "__main__":
    create_small_dataset() 