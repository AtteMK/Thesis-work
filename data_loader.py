import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List
import h5py
from data_preprocessing import DataPreprocessor

class BMESpecimenDataset(Dataset):
    def __init__(self, features: np.ndarray, targets: np.ndarray, transform=None):
        """
        Initialize the dataset.
        
        Args:
            features (np.ndarray): Feature array
            targets (np.ndarray): Target array
            transform: Optional transform to be applied to the data
        """
        self.features = torch.FloatTensor(features)
        
        # Ensure targets are 2D
        if len(targets.shape) == 1:
            targets = targets.reshape(-1, 1)
        self.targets = torch.FloatTensor(targets)
        
        self.transform = transform
        
    def __len__(self) -> int:
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a data sample.
        
        Args:
            idx (int): Index of the sample to get
            
        Returns:
            tuple: (features, target)
        """
        features = self.features[idx]
        target = self.targets[idx]
        
        if self.transform:
            features = self.transform(features)
        
        return features, target

def create_data_loaders(
    data_path: str,
    batch_size: int = 32,
    train_split: float = 0.8,
    random_seed: int = 42
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation data loaders from processed data.
    
    Args:
        data_path (str): Path to the processed HDF5 data file
        batch_size (int): Batch size for the data loaders
        train_split (float): Proportion of data to use for training
        random_seed (int): Random seed for reproducibility
        
    Returns:
        tuple: (train_loader, val_loader)
    """
    # Set random seed for reproducibility
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
    # Load processed data
    preprocessor = DataPreprocessor(data_dir="")  # Empty dir since we're loading processed data
    features, targets = preprocessor.load_processed_data(data_path)
    
    # Prepare training data
    train_data, val_data = preprocessor.prepare_training_data(
        features, targets, train_split, random_seed
    )
    
    # Create datasets
    train_dataset = BMESpecimenDataset(
        features=train_data['features'],
        targets=train_data['targets']
    )
    
    val_dataset = BMESpecimenDataset(
        features=val_data['features'],
        targets=val_data['targets']
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Modify based on your system
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # Modify based on your system
        pin_memory=True
    )
    
    return train_loader, val_loader

def process_and_save_data(
    raw_data_dir: str,
    output_path: str,
    batch_size: int = 32,
    train_split: float = 0.8,
    random_seed: int = 42
) -> Tuple[DataLoader, DataLoader]:
    """
    Process raw data and create data loaders.
    
    Args:
        raw_data_dir (str): Directory containing raw .bmespecimen files
        output_path (str): Path to save processed data
        batch_size (int): Batch size for the data loaders
        train_split (float): Proportion of data to use for training
        random_seed (int): Random seed for reproducibility
        
    Returns:
        tuple: (train_loader, val_loader)
    """
    # Initialize preprocessor
    preprocessor = DataPreprocessor(raw_data_dir)
    
    # Process all files
    features, targets = preprocessor.process_all_files()
    
    # Ensure targets are 2D
    if len(targets.shape) == 1:
        targets = targets.reshape(-1, 1)
    
    # Save processed data
    preprocessor.save_processed_data(features, targets, output_path)
    
    # Create data loaders
    return create_data_loaders(
        output_path,
        batch_size=batch_size,
        train_split=train_split,
        random_seed=random_seed
    ) 