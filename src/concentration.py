import pandas as pd
import json
import numpy as np
import os

# Configuration
DATASHEETS_PATH = '../../data-commons/sensor-datasheets/datasheets.json'
INPUT_CSV = '../../data/smellnet_raw.csv'
OUTPUT_CSV = '../../data/smellnet_eci.csv'

# Approximate mapping from SmellNet columns to OpenSmell sensor models.
# SmellNet's exact hardware is not publicly documented; these are the closest 
# functional equivalents based on the target gases listed in the SmellNet paper.
SMELLNET_MAPPING = {
    'NO2': 'MQ-135',      # MQ-135 is sensitive to NOx
    'C2H5OH': 'MQ-3',     # MQ-3 is highly sensitive to ethanol
    'VOC': 'MQ-135',      # MQ-135 is a broad-spectrum VOC sensor
    'CO': 'MQ-7',         # MQ-7 is selective for Carbon Monoxide
    'Alcohol': 'MQ-3',    # MQ-3 is the standard alcohol sensor
    'LPG': 'MQ-6'         # MQ-6 is highly sensitive to LPG/propane
}

def load_datasheets(path):
    """Load the sensor constants JSON file."""
    with open(path, 'r') as f:
        return json.load(f)

def compute_rs(vout, vcc, r_load):
    """
    Compute sensor resistance Rs from the voltage divider output.
    Circuit: Vcc -> Rs -> Vout -> R_load -> GND
    """
    # Clip to avoid division by zero or negative resistance
    vout = np.clip(vout, 1e-6, vcc - 1e-6)
    return r_load * ((vcc / vout) - 1)

def estimate_r0(rs_series, window=20):
    """
    Estimate baseline resistance R0 from the first N time-steps.
    Assumes the recording starts in clean background air.
    """
    return np.median(rs_series.iloc[:window])

def compute_eci(rs, r0, a, b):
    """
    Compute Estimated Concentration Index (ECI).
    Formula derived from Rs/R0 = a * C^b  =>  C = ( (Rs/R0) / a )^(1/b)
    """
    if a <= 0 or b == 0:
        return np.zeros_like(rs)
    
    ratio = rs / r0
    # Avoid negative base in fractional power or division by zero
    ratio = np.clip(ratio / a, 1e-6, None)
    return ratio ** (1 / b)

def main():
    print("Starting OpenSmell Concentration Module...")
    
    # 1. Load configuration
    datasheets = load_datasheets(DATASHEETS_PATH)
    circuit = datasheets['circuit_defaults']
    vcc = circuit['supply_voltage_v']
    r_load = circuit['load_resistance_ohm']
    
    # Create a lookup for sensor constants
    sensor_constants = {}
    for sensor in datasheets['sensors']:
        model = sensor['sensor_model']
        # Assuming target_gases[0] is the primary calibration gas for the module
        gas_data = sensor['target_gases'][0]
        sensor_constants[model] = {'a': gas_data['a'], 'b': gas_data['b']}
        
    # 2. Load SmellNet data
    if not os.path.exists(INPUT_CSV):
        print(f"Error: Input file {INPUT_CSV} not found.")
        return
        
    df = pd.read_csv(INPUT_CSV)
    eci_df = pd.DataFrame(index=df.index)
    
    # 3. Process each sensor column
    processed_count = 0
    for smell_col, sensor_model in SMELLNET_MAPPING.items():
        if smell_col not in df.columns:
            print(f"Warning: Column '{smell_col}' not found in input CSV. Skipping.")
            continue
            
        if sensor_model not in sensor_constants:
            print(f"Warning: Constants for {sensor_model} not found in datasheets.json. Skipping '{smell_col}'.")
            continue
            
        a = sensor_constants[sensor_model]['a']
        b = sensor_constants[sensor_model]['b']
        
        if a is None or b is None:
            print(f"Warning: Constants for {sensor_model} are null (not yet extracted). Skipping '{smell_col}'.")
            continue

        # Compute Rs (assuming input CSV contains raw voltage Vout)
        vout = df[smell_col]
        rs = compute_rs(vout, vcc, r_load)
        
        # Estimate R0 from the first 20 time-steps
        r0 = estimate_r0(rs)
        
        # Compute ECI
        eci = compute_eci(rs, r0, a, b)
        
        # Store in new dataframe
        eci_df[f"{smell_col}_ECI"] = eci
        processed_count += 1
        print(f"Processed '{smell_col}' -> {sensor_model} (a={a:.3f}, b={b:.4f})")
        
    # 4. Save output
    if processed_count > 0:
        eci_df.to_csv(OUTPUT_CSV, index=False)
        print(f"\nSuccessfully computed ECI for {processed_count} channels.")
        print(f"Saved to {OUTPUT_CSV}")
    else:
        print("\nNo channels were processed. Check your input CSV and datasheets.json.")

if __name__ == "__main__":
    main()