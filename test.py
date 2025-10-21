import numpy as np
import pandas as pd
from typing import Tuple, List, Dict
import os
import h5py
import json
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self, data_dir: str):
        """
        Initialize the data preprocessor.
        
        Args:
            data_dir (str): Directory containing the raw data files
        """
        self.data_dir = data_dir
        self.scaler = StandardScaler()
        self.feature_names = None
        self.target_name = None
        
    def extract_heater_profile_features(self, heater_profiles: List[Dict]) -> np.ndarray:
        """
        Extract features from heater profiles.
        
        Args:
            heater_profiles (List[Dict]): List of heater profile dictionaries
            
        Returns:
            np.ndarray: Array of heater profile features
        """
        features = []
        for profile in heater_profiles:
            # Extract temperature and duration features
            temps = [step['temperature'] for step in profile['steps']]
            durations = [step['duration'] for step in profile['steps']]
            
            # Calculate statistics
            features.extend([
                np.mean(temps),
                np.std(temps),
                np.max(temps),
                np.min(temps),
                np.mean(durations),
                np.std(durations),
                profile['timeBase']
            ])
        
        return np.array(features)
    
    def parse_timestamp(self, timestamp: str) -> float:
        """
        Parse ISO timestamp to Unix timestamp.
        
        Args:
            timestamp (str): ISO format timestamp
            
        Returns:
            float: Unix timestamp
        """
        return datetime.fromisoformat(timestamp.replace('Z', '+00:00')).timestamp()
    
    def read_bmespecimen_file(self, file_path: str) -> Tuple[np.ndarray, str]:
        """
        Read and parse a .bmespecimen file.
        
        Args:
            file_path (str): Path to the .bmespecimen file
            
        Returns:
            tuple: (features, target)
        """
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
                # Extract relevant data from the JSON structure
                specimen_data = data['data']['specimenData']
                specimen_data_poits = data['data']['specimenDataPoints']
                heater_profiles = data['data']['heaterProfiles']
                
                # Extract basic features
                #basic_features = np.array([
                #    specimen_data_poits[1],
                #    specimen_data_poits[2],
                #    specimen_data_poits[3],
                #    specimen_data_poits[4]
                #])
                features = np.array([
                    specimen_data_poits[1],
                    specimen_data_poits[2],
                    specimen_data_poits[3],
                    specimen_data_poits[4]
                ])
                
                # Extract heater profile features
                heater_features = self.extract_heater_profile_features(heater_profiles)
                
                # Combine all features
                #features = np.concatenate([basic_features, heater_features])
                
                # Target is now a string (specimen ID or other identifier)
                target = str(specimen_data['id'])
                
                return features, target
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {str(e)}")
            raise
    
    def clean_data(self, features: np.ndarray, target: str) -> Tuple[np.ndarray, str]:
        """
        Clean the data by removing outliers and handling missing values.
        
        Args:
            features (np.ndarray): Feature array
            target (str): Target string
            
        Returns:
            tuple: (cleaned_features, cleaned_target)
        """
        # Remove outliers using IQR method
        Q1 = np.percentile(features, 25)
        Q3 = np.percentile(features, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Replace outliers with median
        features_cleaned = np.clip(features, lower_bound, upper_bound)
        
        # Handle any NaN or infinite values
        features_cleaned = np.nan_to_num(features_cleaned, nan=np.nanmedian(features_cleaned))
        
        return features_cleaned, target
    
    def process_all_files(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process all .bmespecimen files in the directory.
        
        Returns:
            tuple: (features_array, targets_array)
        """
        import glob
        
        all_features = []
        all_targets = []
        
        file_paths = glob.glob(os.path.join(self.data_dir, "*.bmespecimen"))
        logger.info(f"Found {len(file_paths)} files to process")
        
        for file_path in file_paths:
            try:
                features, target = self.read_bmespecimen_file(file_path)
                features_cleaned, target_cleaned = self.clean_data(features, target)
                
                all_features.append(features_cleaned)
                all_targets.append(target_cleaned)
            except Exception as e:
                logger.warning(f"Skipping file {file_path} due to error: {str(e)}")
                continue
        
        if not all_features:
            raise ValueError("No valid data files were processed")
            
        return np.array(all_features), np.array(all_targets, dtype='U')
    
    def save_processed_data(self, 
                          features: np.ndarray, 
                          targets: np.ndarray, 
                          output_path: str):
        """
        Save processed data to an HDF5 file.
        
        Args:
            features (np.ndarray): Processed features
            targets (np.ndarray): Processed string targets
            output_path (str): Path to save the HDF5 file
        """
        with h5py.File(output_path, 'w') as f:
            f.create_dataset('features', data=features)
            
            # Store string targets properly
            string_dt = h5py.string_dtype(encoding='utf-8')
            f.create_dataset('targets', data=targets, dtype=string_dt)
            
            # Save metadata
            metadata = {
                'num_samples': len(features),
                'feature_dim': features.shape[1],
                'target_type': 'string'
            }
            for key, value in metadata.items():
                f.attrs[key] = value
    
    def load_processed_data(self, file_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load processed data from an HDF5 file.
        
        Args:
            file_path (str): Path to the HDF5 file
            
        Returns:
            tuple: (features, targets)
        """
        with h5py.File(file_path, 'r') as f:
            features = f['features'][:]
            targets = f['targets'][:].astype(str)
        return features, targets
    
    def prepare_training_data(self, 
                            features: np.ndarray, 
                            targets: np.ndarray,
                            train_split: float = 0.8,
                            random_seed: int = 42) -> Dict[str, np.ndarray]:
        """
        Prepare data for training by splitting into train/val sets and scaling.
        
        Args:
            features (np.ndarray): Feature array
            targets (np.ndarray): Target array (string labels)
            train_split (float): Proportion of data to use for training
            random_seed (int): Random seed for reproducibility
            
        Returns:
            dict: Dictionary containing train and validation sets
        """
        np.random.seed(random_seed)
        
        indices = np.random.permutation(len(features))
        split_idx = int(len(features) * train_split)
        
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]
        
        # Scale features only
        features_scaled = self.scaler.fit_transform(features)
        
        train_data = {
            'features': features_scaled[train_indices],
            'targets': targets[train_indices]
        }
        
        val_data = {
            'features': features_scaled[val_indices],
            'targets': targets[val_indices]
        }
        
        return train_data, val_data


if __name__ == "__main__":
    data_dir = "./data/specimendata"  # Change this to your data folder
    output_file = "./processed_data_test.h5"
    
    preprocessor = DataPreprocessor(data_dir)
    features, targets = preprocessor.process_all_files()
    preprocessor.save_processed_data(features, targets, output_file)
    
    logger.info(f"Processed data saved to {output_file}")
