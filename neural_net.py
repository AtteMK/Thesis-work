import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple, List
import matplotlib.pyplot as plt

class NeuralNetwork(nn.Module):
    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int):
        """
        Initialize the neural network.
        
        Args:
            input_size (int): Number of input features
            hidden_sizes (List[int]): List of hidden layer sizes
            output_size (int): Number of output features
        """
        super(NeuralNetwork, self).__init__()
        
        # Create a list to hold all layers
        layers = []
        
        # Input layer
        prev_size = input_size
        
        # Add hidden layers
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Dropout(0.2)
            ])
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, output_size))
        
        # Combine all layers
        self.model = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        return self.model(x)

class ModelTrainer:
    def __init__(self, model: nn.Module, learning_rate: float = 0.001):
        """
        Initialize the model trainer.
        
        Args:
            model (nn.Module): Neural network model
            learning_rate (float): Learning rate for optimization
        """
        self.model = model
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.train_losses = []
        self.val_losses = []
        
    def train_step(self, X: torch.Tensor, y: torch.Tensor) -> float:
        """Perform one training step."""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        outputs = self.model(X)
        loss = self.criterion(outputs, y)
        
        # Backward pass and optimize
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def validate(self, X: torch.Tensor, y: torch.Tensor) -> float:
        """Perform validation."""
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X)
            loss = self.criterion(outputs, y)
        return loss.item()
    
    def train(self, 
              train_loader: torch.utils.data.DataLoader,
              val_loader: torch.utils.data.DataLoader,
              epochs: int) -> Tuple[List[float], List[float]]:
        """
        Train the model.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            epochs: Number of training epochs
        
        Returns:
            Tuple of training and validation losses
        """
        for epoch in range(epochs):
            epoch_train_losses = []
            epoch_val_losses = []
            
            # Training
            for batch_X, batch_y in train_loader:
                train_loss = self.train_step(batch_X, batch_y)
                epoch_train_losses.append(train_loss)
            
            # Validation
            for batch_X, batch_y in val_loader:
                val_loss = self.validate(batch_X, batch_y)
                epoch_val_losses.append(val_loss)
            
            # Calculate average losses
            avg_train_loss = np.mean(epoch_train_losses)
            avg_val_loss = np.mean(epoch_val_losses)
            
            self.train_losses.append(avg_train_loss)
            self.val_losses.append(avg_val_loss)
            
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'Training Loss: {avg_train_loss:.4f}')
            print(f'Validation Loss: {avg_val_loss:.4f}')
        
        return self.train_losses, self.val_losses
    
    def plot_losses(self):
        """Plot training and validation losses."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Losses')
        plt.legend()
        plt.grid(True)
        plt.show()

    def save_model(self, path: str):
        """Save the model to a file."""
        torch.save(self.model.state_dict(), path)
    
    def load_model(self, path: str):
        """Load the model from a file."""
        self.model.load_state_dict(torch.load(path))
        self.model.eval() 