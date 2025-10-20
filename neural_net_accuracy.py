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
        self.criterion = nn.CrossEntropyLoss()  # For classification
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []

    @staticmethod
    def compute_accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> float:
        """
        Compute accuracy given model outputs and true labels.

        Args:
            outputs: Raw model outputs (logits)
            targets: Ground truth labels

        Returns:
            Accuracy as a float
        """
        preds = torch.argmax(outputs, dim=1)
        correct = (preds == targets).sum().item()
        total = targets.size(0)
        return correct / total

    def train_step(self, X: torch.Tensor, y: torch.Tensor) -> Tuple[float, float]:
        """Perform one training step and return loss and accuracy."""
        self.model.train()
        self.optimizer.zero_grad()

        outputs = self.model(X)
        loss = self.criterion(outputs, y)

        loss.backward()
        self.optimizer.step()

        acc = self.compute_accuracy(outputs, y)
        return loss.item(), acc

    def validate(self, X: torch.Tensor, y: torch.Tensor) -> Tuple[float, float]:
        """Perform validation and return loss and accuracy."""
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X)
            loss = self.criterion(outputs, y)
            acc = self.compute_accuracy(outputs, y)
        return loss.item(), acc

    def train(self, train_loader: torch.utils.data.DataLoader,
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
            epoch_train_accuracies = []
            epoch_val_accuracies = []

            for batch_X, batch_y in train_loader:
                print(f'batch_X shape: {batch_X.shape}, dtype: {batch_X.dtype}')
                print(f'batch_y shape: {batch_y.shape}, dtype: {batch_y.dtype}, unique values: {batch_y.unique()}')
                break  # only check first batch

            # Training
            for batch_X, batch_y in train_loader:
                train_loss, train_acc = self.train_step(batch_X, batch_y)
                epoch_train_losses.append(train_loss)
                epoch_train_accuracies.append(train_acc)

            # Validation
            for batch_X, batch_y in val_loader:
                val_loss, val_acc = self.validate(batch_X, batch_y)
                epoch_val_losses.append(val_loss)
                epoch_val_accuracies.append(val_acc)

            avg_train_loss = np.mean(epoch_train_losses)
            avg_val_loss = np.mean(epoch_val_losses)
            avg_train_acc = np.mean(epoch_train_accuracies)
            avg_val_acc = np.mean(epoch_val_accuracies)

            self.train_losses.append(avg_train_loss)
            self.val_losses.append(avg_val_loss)
            self.train_accuracies.append(avg_train_acc)
            self.val_accuracies.append(avg_val_acc)

            print(f'Epoch {epoch+1}/{epochs}')
            print(f'Train Loss: {avg_train_loss:.4f}, Accuracy: {avg_train_acc*100:.2f}%')
            print(f'Val   Loss: {avg_val_loss:.4f}, Accuracy: {avg_val_acc*100:.2f}%')

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

    def plot_accuracy(self):
        """Plot training and validation accuracy."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_accuracies, label='Training Accuracy')
        plt.plot(self.val_accuracies, label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracy')
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