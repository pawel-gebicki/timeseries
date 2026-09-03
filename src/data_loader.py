import pandas as pd
import os

def load_data():
    """
    Loads all project datasets from the data folder.
    Returns a dictionary of dataframes.
    """
    # This automatically finds the path to the 'data' folder
    base_path = os.path.join(os.path.dirname(__file__), '..', 'data')
    
    files = {
        'timeseries': 'timeseries.csv',
        'oil': 'oil.csv',
        'holidays': 'holidays.csv',
        'stores': 'stores.csv'
    }
    
    data = {}
    for key, filename in files.items():
        filepath = os.path.join(base_path, filename)
        if os.path.exists(filepath):
            data[key] = pd.read_csv(filepath)
            print(f"Loaded {key} successfully.")
        else:
            print(f"Error: Could not find {filename}")
            
    return data