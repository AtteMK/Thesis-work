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
        
    def extract_bmespecimen_data(file_path: str) -> Tuple[np.ndarray, float]:
        """
        Extracts structured data from a .bmespecimen JSON file.
        
        Args:
            file_path (str): Path to the .bmespecimen file
        
        Returns:
            dict: Processed JSON data containing selected fields
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to read {file_path}: {e}")
            raise

        try:
            root = data.get("data", {})
            
            # --- specimenData ---
            specimen_data = root.get("specimenData", {})
            specimen_info = {
                "id": specimen_data.get("id"),
                "label": specimen_data.get("label")
            }

            # --- heaterProfiles ---
            heater_profiles = []
            for hp in root.get("heaterProfiles", []):
                steps = hp.get("steps", [])
                temperatures = [s.get("temperature") for s in steps if "temperature" in s]
                heater_profiles.append({
                    "id": hp.get("id"),
                    "uid": hp.get("uid"),
                    "temperatures": temperatures
                })

            # --- sensorConfigs ---
            sensor_configs = []
            for sc in root.get("sensorConfigs", []):
                sensor_configs.append({
                    "id": sc.get("id"),
                    "sensorId": sc.get("sensorId"),
                    "heaterProfileId": sc.get("heaterProfileId")
                })

            # --- cycles ---
            cycles = []
            for cy in root.get("cycles", []):
                cycles.append({
                    "id": cy.get("id"),
                    "sensorId": cy.get("sensorId")
                })

            # --- dataColumns ---
            data_columns = []
            for dc in root.get("dataColumns", []):
                data_columns.append({
                    "name": dc.get("name"),
                    "key": dc.get("key")
                })

            # Extract only the list of keys for mapping
            column_keys = [dc.get("key") for dc in data_columns if "key" in dc]

            # --- specimenDataPoints ---
            specimen_data_points = root.get("specimenDataPoints", [])

            # Map data points to their corresponding column keys
            mapped_data_points = []

            # Handle case: multiple data points (list of lists)
            if specimen_data_points and isinstance(specimen_data_points[0], list):
                for point in specimen_data_points:
                    point_map = {}
                    for idx, key in enumerate(column_keys):
                        if idx < len(point):
                            point_map[key] = point[idx]
                    mapped_data_points.append(point_map)
            else:
                # Single list of values
                point_map = {}
                for idx, key in enumerate(column_keys):
                    if idx < len(specimen_data_points):
                        point_map[key] = specimen_data_points[idx]
                mapped_data_points = [point_map]

            # --- Link specimenDataPoints to cycles by cycle_id ---
            if cycles and mapped_data_points:
                # Create a lookup dictionary for fast access
                cycle_lookup = {c["id"]: c["sensorId"] for c in cycles if "id" in c and "sensorId" in c}

                for dp in mapped_data_points:
                    cycle_id = dp.get("cycle_id")  # match key
                    if cycle_id in cycle_lookup:
                        dp["sensor_id"] = cycle_lookup[cycle_id]
            
            # --- Link specimenDataPoints to sensor_configs by sensor_id ---
            if sensor_configs and mapped_data_points:
                # Create a lookup dictionary for fast access
                sensor_configs_lookup = {c["sensorId"]: c["heaterProfileId"] for c in sensor_configs if "sensorId" in c and "heaterProfileId" in c}

                for dp in mapped_data_points:
                    sensor_id = dp.get("sensor_id")  # match key
                    if sensor_id in sensor_configs_lookup:
                        dp["heater_profile_id"] = sensor_configs_lookup[sensor_id]
            
            # --- Link specimenDataPoints to sensor_configs by sensor_id ---
            if heater_profiles and mapped_data_points:
                # Create a lookup dictionary for fast access
                heater_profiles_lookup = {c["id"]: c["temperatures"] for c in heater_profiles if "id" in c and "temperatures" in c}

                for dp in mapped_data_points:
                    index = dp.get("cycle_step_index")
                    heater_profile_id = dp.get("heater_profile_id")  # match key
                    if heater_profile_id in heater_profiles_lookup:
                        dp["heater_temperature"] = heater_profiles_lookup[heater_profile_id][index]

            # --- build final structured output ---
            extracted_data = {
                "specimenDataPoints": mapped_data_points
            }

            features = np.array([
                mapped_data_points['resistance_gassensor'],
                mapped_data_points['temperature'],
                mapped_data_points['pressure'],
                mapped_data_points['relative_humidity'],
                mapped_data_points['specimen_id'],
                mapped_data_points['heater_temperature']
                ])

            target = mapped_data_points['specimen_id']

            return features, target
        
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            raise


    def save_processed_data(self, 
                          features: np.ndarray, 
                          targets: np.ndarray, 
                          output_path: str):
        """
        Save processed data to an HDF5 file.
        
        Args:
            features (np.ndarray): Processed features
            targets (np.ndarray): Processed targets
            output_path (str): Path to save the HDF5 file
        """
        with h5py.File(output_path, 'w') as f:
            f.create_dataset('features', data=features)
            f.create_dataset('targets', data=targets)
            
            # Save metadata
            metadata = {
                'num_samples': len(features),
                'feature_dim': features.shape[1],
                'target_dim': targets.shape[1] if len(targets.shape) > 1 else 1
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
            targets = f['targets'][:]
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
            targets (np.ndarray): Target array
            train_split (float): Proportion of data to use for training
            random_seed (int): Random seed for reproducibility
            
        Returns:
            dict: Dictionary containing train and validation sets
        """
        # Set random seed
        np.random.seed(random_seed)
        
        # Split data
        indices = np.random.permutation(len(features))
        split_idx = int(len(features) * train_split)
        
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        # Create train and validation sets
        train_data = {
            'features': features_scaled[train_indices],
            'targets': targets[train_indices]
        }
        
        val_data = {
            'features': features_scaled[val_indices],
            'targets': targets[val_indices]
        }
        
        return train_data, val_data 