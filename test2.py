import json
import os
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_bmespecimen_data(file_path: str) -> dict:
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
            "specimenData": specimen_info,
            "heaterProfiles": heater_profiles,
            "sensorConfigs": sensor_configs,
            "cycles": cycles,
            "dataColumns": data_columns,
            "specimenDataPoints": mapped_data_points
        }

        return extracted_data
    
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {e}")
        raise


def save_processed_json(data: dict, input_path: str):
    """
    Save processed data as a JSON file with '_processed' suffix.
    
    Args:
        data (dict): Processed data dictionary
        input_path (str): Original file path
    """
    base, ext = os.path.splitext(input_path)
    output_path = base + "_processed.json"

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        logger.info(f"✅ Processed data saved to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save processed JSON: {e}")
        raise


def main():
    if len(sys.argv) < 2:
        print("Usage: python process_bmespecimen.py <path_to_file.bmespecimen>")
        sys.exit(1)

    file_path = sys.argv[1]

    if not os.path.isfile(file_path):
        print(f"❌ File not found: {file_path}")
        sys.exit(1)

    logger.info(f"Processing file: {file_path}")

    extracted_data = extract_bmespecimen_data(file_path)
    save_processed_json(extracted_data, file_path)


if __name__ == "__main__":
    main()
