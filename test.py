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

    def read_bmespecimen_file(self, file_path: str) -> Tuple[np.ndarray, float]:
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
                measurement_session = data['data']['measurementSession']
                board_config = data['data']['boardConfig']
                board_type = data['data']['boardType']
                heater_profiles = data['data']['heaterProfiles']
                specimen_data_points = data['data']['specimenDataPoints'][1,2,3,4]
                
                # Extract basic features
                basic_features = np.array([
                    specimen_data['id'],
                    specimen_data['startTime'],
                    specimen_data['endTime'],
                    self.parse_timestamp(specimen_data['createdAt']),
                    self.parse_timestamp(specimen_data['updatedAt']),
                    self.parse_timestamp(measurement_session['startTime']),
                    self.parse_timestamp(measurement_session['endTime']),
                    board_type['numSensors']
                ])
                
                # Extract heater profile features
                heater_features = self.extract_heater_profile_features(heater_profiles)
                
                # Combine all features
                features = np.concatenate([basic_features, heater_features])
                
                # Extract target (modify based on your needs)
                # For now, we'll use the specimen ID as a placeholder target
                target = specimen_data['id']
                
                print(features, target)
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {str(e)}")
            raise

if __name__ == "__main__":
    raw_data_dir = r"C:\Users\atte0\Documents\Thesis work\data\specimendata\laboratory_water_ethanol_5_v_alc_6_v_34.bmespecimen"
    preprocessor = DataPreprocessor(raw_data_dir)
    features, targets = preprocessor.process_all_files()
