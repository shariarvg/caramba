import torch
import torch.nn as nn
import numpy as np
from dataset import TokenizedMovieDataset
from reshape_movies import get_dictionary_of_movies
from inference import load_embedding_model_and_tokenizer, obtain_double_hidden_states
from train_predictor import TokenPredictor
from sklearn.linear_model import LogisticRegression
from transformers import AutoModelForCausalLM

def load_predictors(vocab_size, input_dim):
    """Load both the gradient descent and logistic regression predictors."""
    # Load gradient descent model
    gd_model = TokenPredictor(input_dim, vocab_size)
    gd_model.load_state_dict(torch.load('token_predictor_gd.pt'))
    gd_model.eval()
    
    # Load logistic regression model
    lr_model = np.load('token_predictor_lr.npy', allow_pickle=True).item()
    
    return gd_model, lr_model

def evaluate_gpt2(model, tokenizer, dataset, batch_size=32):
    """Evaluate GPT-2's performance on predicting the last token."""
    print("Evaluating GPT-2...")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in dataloader:
            actual = batch['input_ids'][:, -1]
            input = batch['input_ids'][:, :-1].to(model.device)
            attention_mask = torch.ones_like(input).to(model.device)
            
            # Get model predictions
            outputs = model(input, attention_mask=attention_mask)
            # Use the last position's logits to predict the next token
            next_token_logits = outputs.logits[:, -1, :]
            predicted = torch.argmax(next_token_logits, dim=-1)
            
            # Compare with actual last tokens
            correct += (predicted == actual).sum().item()
            total += actual.size(0)
    
    accuracy = 100 * correct / total
    print(f"GPT-2 Accuracy: {accuracy:.2f}%")
    return accuracy

def evaluate_predictors(gd_model, lr_model, hidden_states, labels):
    """Evaluate both predictor models."""
    print("Evaluating predictors...")
    
    # Prepare data
    hidden_states_pooled = hidden_states.mean(dim=1)
    
    # Evaluate gradient descent model
    with torch.no_grad():
        outputs = gd_model(hidden_states_pooled)
        _, predicted = torch.max(outputs.data, 1)
        gd_correct = (predicted == labels).sum().item()
        gd_accuracy = 100 * gd_correct / len(labels)
    
    # Evaluate logistic regression model
    lr_predictions = lr_model.predict(hidden_states_pooled.numpy())
    lr_correct = (lr_predictions == labels.numpy()).sum()
    lr_accuracy = 100 * lr_correct / len(labels)
    
    print(f"Gradient Descent Predictor Accuracy: {gd_accuracy:.2f}%")
    print(f"Logistic Regression Predictor Accuracy: {lr_accuracy:.2f}%")
    
    return gd_accuracy, lr_accuracy

def main():
    # Load model and tokenizer
    model, tokenizer = load_embedding_model_and_tokenizer("gpt2")
    
    # Load held-out dataset (last 1000 movies)
    print("Loading held-out dataset...")
    all_movies = get_dictionary_of_movies(N_MOVIES_TO_USE=100, start_movie_id=2000)  # Load more than we need
    held_out_dataset = TokenizedMovieDataset(all_movies, tokenizer, padding=True, max_length=512)
    
    # Get hidden states for the held-out set
    print("Obtaining hidden states for held-out set...")
    hidden_states = obtain_double_hidden_states(model, tokenizer, held_out_dataset)
    hidden_states = torch.from_numpy(hidden_states).float()
    
    # Get labels (last tokens)
    labels = torch.tensor([item['input_ids'][-1].item() for item in held_out_dataset])
    
    # Load predictors
    input_dim = hidden_states.shape[-1]  # 2D
    gd_model, lr_model = load_predictors(tokenizer.vocab_size, input_dim)
    
    # Load GPT-2 model for evaluation
    gpt2_model = AutoModelForCausalLM.from_pretrained("gpt2")
    gpt2_model.eval()
    
    # Evaluate all models
    gpt2_accuracy = evaluate_gpt2(gpt2_model, tokenizer, held_out_dataset)
    gd_accuracy, lr_accuracy = evaluate_predictors(gd_model, lr_model, hidden_states, labels)
    
    # Print comparison
    print("\nModel Comparison:")
    print(f"GPT-2: {gpt2_accuracy:.2f}%")
    print(f"Gradient Descent Predictor: {gd_accuracy:.2f}%")
    print(f"Logistic Regression Predictor: {lr_accuracy:.2f}%")

if __name__ == "__main__":
    main() 