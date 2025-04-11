import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import LogisticRegression
import numpy as np
from dataset import TokenizedMovieDataset
from reshape_movies import get_dictionary_of_movies
from inference import load_embedding_model_and_tokenizer, obtain_double_hidden_states_and_labels
import sys



class TokenPredictor(nn.Module):
    """
    A simple neural network for predicting the next token.
    
    This model takes hidden states as input and predicts the next token in the sequence.
    It consists of a single linear layer that maps from the hidden state dimension
    to the vocabulary size.
    
    Attributes:
        linear (nn.Linear): Linear layer for token prediction
    """
    
    def __init__(self, input_dim, vocab_size):
        """
        Initialize the token predictor.
        
        Args:
            input_dim (int): Dimension of the input hidden states
            vocab_size (int): Size of the vocabulary (number of possible tokens)
        """
        super().__init__()
        self.linear = nn.Linear(input_dim, vocab_size)
        
    def forward(self, x):
        """
        Forward pass through the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, vocab_size)
        """
        return self.linear(x)

def prepare_data(model_name):
    """
    Prepare the dataset for training by loading hidden states and labels.
    
    This function loads the hidden states and labels from saved files.
    
    Args:
        model_name (str): Name of the model used to generate the hidden states
        
    Returns:
        tuple: (hidden_states, labels) - numpy arrays of hidden states and labels
    """
    print("Loading hidden states...")
    hidden_states = np.load(f"double_hidden_states_{model_name}.npy")
    
    print("Loading labels...")
    labels = np.load(f"labels_{model_name}.npy")
    
    return hidden_states, labels

def train_gradient_descent(hidden_states_pooled, labels, vocab_size, n_epochs=100, batch_size=16):
    """
    Train a neural network using gradient descent.
    
    This function trains a TokenPredictor model using the Adam optimizer
    and cross-entropy loss. It prints the loss and accuracy for each epoch.
    
    Args:
        hidden_states_pooled (numpy.ndarray): Hidden states with shape (N, D)
        labels (numpy.ndarray): Labels with shape (N,)
        vocab_size (int): Size of the vocabulary
        n_epochs (int, optional): Number of training epochs. Defaults to 10.
        batch_size (int, optional): Batch size for training. Defaults to 16.
        
    Returns:
        TokenPredictor: The trained model
    """
    print("Training with gradient descent...")
    
    # Initialize model and move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Convert numpy arrays to PyTorch tensors
    hidden_states_tensor = torch.tensor(hidden_states_pooled, dtype=torch.float16)
    labels_tensor = torch.tensor(labels, dtype=torch.foat16)
    
    model = TokenPredictor(hidden_states_tensor.shape[-1], vocab_size).to(device)
    
    # Create data loader
    dataset = torch.utils.data.TensorDataset(hidden_states_tensor, labels_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_hidden, batch_labels in dataloader:
            batch_hidden = batch_hidden.to(device)
            batch_labels = batch_labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_hidden)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()
        
        mean_accuracy = 100 * correct / total
        print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {total_loss/len(dataloader):.4f}, Accuracy: {mean_accuracy:.2f}%')
    
    return model

def train_logistic_regression(hidden_states, labels, vocab_size):
    """
    Train a logistic regression model.
    
    This function trains a multinomial logistic regression model using scikit-learn.
    It prints the accuracy of the model on the training data.
    
    Args:
        hidden_states (numpy.ndarray): Hidden states with shape (N, D)
        labels (numpy.ndarray): Labels with shape (N,)
        vocab_size (int): Size of the vocabulary
        
    Returns:
        LogisticRegression: The trained model
    """
    print("Training with logistic regression...")
    
    # Mean pool across sequence length dimension
    X = hidden_states  # Shape: (batch_size, 2D)
    y = labels#.numpy()
    
    # Train logistic regression
    model = LogisticRegression(multi_class='multinomial', max_iter=1000)
    model.fit(X, y)
    
    # Calculate accuracy
    accuracy = model.score(X, y) * 100
    print(f'Logistic Regression Accuracy: {accuracy:.2f}%')
    
    return model

def main():
    """
    Main function to train token predictors.
    
    This function:
    1. Loads hidden states and labels
    2. Trains a neural network using gradient descent
    3. Trains a logistic regression model
    4. Saves both models
    
    Command line arguments:
        sys.argv[1]: model_name - The name of the model used to generate the hidden states
    """
    model_name = sys.argv[1]
    # Load model and tokenizer
    _, tokenizer = load_embedding_model_and_tokenizer(model_name)
    # Prepare data
    hidden_states, labels = prepare_data(model_name)
    
    # Train models
    vocab_size = tokenizer.vocab_size
    
    # Train with gradient descent
    gd_model = train_gradient_descent(hidden_states, labels, vocab_size)
    
    # Train with logistic regression
    lr_model = train_logistic_regression(hidden_states, labels, vocab_size)
    
    # Save models
    torch.save(gd_model.state_dict(), f'../token_predictor_{model_name}_gd.pt')
    np.save(f'../token_predictor_{model_name}_lr.npy', lr_model)

if __name__ == "__main__":
    main()