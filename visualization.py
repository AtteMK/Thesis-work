import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import pandas as pd
from sklearn.decomposition import PCA
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataVisualizer:
    def __init__(self, features: np.ndarray, targets: np.ndarray):
        """
        Initialize the data visualizer.
        
        Args:
            features (np.ndarray): Feature array
            targets (np.ndarray): Target array
        """
        self.features = features
        self.targets = targets
        self.feature_names = [f"Feature_{i}" for i in range(features.shape[1])]
        self.target_names = [f"Target_{i}" for i in range(targets.shape[1])]
        
    def plot_feature_distributions(self, save_path: str = None):
        """
        Plot distributions of all features.
        
        Args:
            save_path (str, optional): Path to save the plot
        """
        n_features = self.features.shape[1]
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        plt.figure(figsize=(15, 5 * n_rows))
        
        for i in range(n_features):
            plt.subplot(n_rows, n_cols, i + 1)
            sns.histplot(self.features[:, i], kde=True)
            plt.title(f"Distribution of {self.feature_names[i]}")
            plt.xlabel("Value")
            plt.ylabel("Count")
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    def plot_correlation_matrix(self, save_path: str = None):
        """
        Plot correlation matrix of features.
        
        Args:
            save_path (str, optional): Path to save the plot
        """
        # Create DataFrame for correlation analysis
        df = pd.DataFrame(self.features, columns=self.feature_names)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm', center=0)
        plt.title("Feature Correlation Matrix")
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    def plot_pca_analysis(self, n_components: int = 2, save_path: str = None):
        """
        Perform PCA analysis and plot results.
        
        Args:
            n_components (int): Number of PCA components to use
            save_path (str, optional): Path to save the plot
        """
        pca = PCA(n_components=n_components)
        features_pca = pca.fit_transform(self.features)
        
        # Plot explained variance ratio
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(range(1, len(pca.explained_variance_ratio_) + 1),
                np.cumsum(pca.explained_variance_ratio_))
        plt.xlabel("Number of Components")
        plt.ylabel("Cumulative Explained Variance Ratio")
        plt.title("PCA Explained Variance")
        
        # Plot first two components
        plt.subplot(1, 2, 2)
        plt.scatter(features_pca[:, 0], features_pca[:, 1], c=self.targets.ravel())
        plt.xlabel("First Principal Component")
        plt.ylabel("Second Principal Component")
        plt.title("PCA Components")
        plt.colorbar(label="Target Value")
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    def plot_feature_target_correlations(self, save_path: str = None):
        """
        Plot correlations between features and targets.
        
        Args:
            save_path (str, optional): Path to save the plot
        """
        # Create DataFrame for correlation analysis
        df_features = pd.DataFrame(self.features, columns=self.feature_names)
        df_targets = pd.DataFrame(self.targets, columns=self.target_names)
        
        # Calculate correlations
        correlations = pd.DataFrame(index=self.feature_names)
        for target in self.target_names:
            correlations[target] = [df_features[feature].corr(df_targets[target])
                                  for feature in self.feature_names]
        
        # Plot correlations
        plt.figure(figsize=(10, 6))
        sns.heatmap(correlations, annot=True, cmap='coolwarm', center=0)
        plt.title("Feature-Target Correlations")
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    def plot_training_data_split(self, train_indices: np.ndarray, val_indices: np.ndarray,
                               save_path: str = None):
        """
        Visualize the training/validation split of the data.
        
        Args:
            train_indices (np.ndarray): Indices of training samples
            val_indices (np.ndarray): Indices of validation samples
            save_path (str, optional): Path to save the plot
        """
        # Perform PCA for visualization
        pca = PCA(n_components=2)
        features_pca = pca.fit_transform(self.features)
        
        plt.figure(figsize=(10, 6))
        plt.scatter(features_pca[train_indices, 0], features_pca[train_indices, 1],
                   label='Training', alpha=0.6)
        plt.scatter(features_pca[val_indices, 0], features_pca[val_indices, 1],
                   label='Validation', alpha=0.6)
        plt.xlabel("First Principal Component")
        plt.ylabel("Second Principal Component")
        plt.title("Training/Validation Split")
        plt.legend()
        
        if save_path:
            plt.savefig(save_path)
        plt.show()

def visualize_processed_data(data_path: str, output_dir: str = "visualizations"):
    """
    Create visualizations for processed data.
    
    Args:
        data_path (str): Path to the processed HDF5 data file
        output_dir (str): Directory to save visualization plots
    """
    import os
    import h5py
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    with h5py.File(data_path, 'r') as f:
        features = f['features'][:]
        targets = f['targets'][:]
    
    # Create visualizer
    visualizer = DataVisualizer(features, targets)
    
    # Generate visualizations
    logger.info("Generating feature distributions...")
    visualizer.plot_feature_distributions(os.path.join(output_dir, "feature_distributions.png"))
    
    logger.info("Generating correlation matrix...")
    visualizer.plot_correlation_matrix(os.path.join(output_dir, "correlation_matrix.png"))
    
    logger.info("Generating PCA analysis...")
    visualizer.plot_pca_analysis(save_path=os.path.join(output_dir, "pca_analysis.png"))
    
    logger.info("Generating feature-target correlations...")
    visualizer.plot_feature_target_correlations(os.path.join(output_dir, "feature_target_correlations.png"))
    
    logger.info(f"Visualizations saved in {output_dir}") 