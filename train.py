import torch
from neural_net import NeuralNetwork, ModelTrainer
from data_loader import process_and_save_data, create_data_loaders
from visualization import visualize_processed_data
import argparse
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train neural network on BME specimen data')
    parser.add_argument('--data_dir', type=str, default='.', help='Directory containing .bmespecimen files')
    parser.add_argument('--processed_data_path', type=str, default='processed_data.h5', help='Path to save/load processed data')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden_sizes', type=str, default='512,256,128', help='Comma-separated list of hidden layer sizes')
    parser.add_argument('--train_split', type=float, default=0.8, help='Proportion of data to use for training')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--model_save_path', type=str, default='trained_model.pth', help='Path to save the trained model')
    parser.add_argument('--load_processed', action='store_true', help='Load existing processed data instead of processing raw data')
    parser.add_argument('--visualize', action='store_true', help='Generate data visualizations')
    parser.add_argument('--visualization_dir', type=str, default='visualizations', help='Directory to save visualizations')
    args = parser.parse_args()

    # Convert hidden_sizes string to list of integers
    hidden_sizes = [int(size) for size in args.hidden_sizes.split(',')]

    # Set random seeds for reproducibility
    torch.manual_seed(args.random_seed)
    import numpy as np
    np.random.seed(args.random_seed)

    # Process data or load processed data
    if args.load_processed and os.path.exists(args.processed_data_path):
        logger.info("Loading existing processed data...")
        train_loader, val_loader = create_data_loaders(
            data_path=args.processed_data_path,
            batch_size=args.batch_size,
            train_split=args.train_split,
            random_seed=args.random_seed
        )
    else:
        logger.info("Processing raw data...")
        train_loader, val_loader = process_and_save_data(
            raw_data_dir=args.data_dir,
            output_path=args.processed_data_path,
            batch_size=args.batch_size,
            train_split=args.train_split,
            random_seed=args.random_seed
        )

    # Generate visualizations if requested
    if args.visualize:
        logger.info("Generating data visualizations...")
        visualize_processed_data(args.processed_data_path, args.visualization_dir)

    # Get input and output sizes from the first batch
    sample_batch = next(iter(train_loader))
    input_size = sample_batch[0].shape[1]
    
    # Handle target dimensions
    if len(sample_batch[1].shape) == 1:
        output_size = 1  # Single target value
    else:
        output_size = sample_batch[1].shape[1]  # Multiple target values

    logger.info(f"Input size: {input_size}, Output size: {output_size}")

    # Initialize model
    model = NeuralNetwork(
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        output_size=output_size
    )

    # Initialize trainer
    trainer = ModelTrainer(model, learning_rate=args.learning_rate)

    # Train the model
    logger.info("Starting training...")
    train_losses, val_losses = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs
    )

    # Plot training results
    trainer.plot_losses()

    # Save the trained model
    trainer.save_model(args.model_save_path)
    logger.info(f"Training completed. Model saved as '{args.model_save_path}'")

if __name__ == "__main__":
    main() 