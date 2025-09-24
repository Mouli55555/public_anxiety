# location_extractor.py
# This module handles adding and processing location data for mapping.

import pandas as pd
import random

# A list of sample cities with their approximate latitude and longitude
# In a real-world scenario, this would come from the tweets themselves.
SAMPLE_LOCATIONS = {
    'New York': (40.7128, -74.0060),
    'London': (51.5074, -0.1278),
    'Tokyo': (35.6895, 139.6917),
    'Sydney': (-33.8688, 151.2093),
    'Mumbai': (19.0760, 72.8777),
    'Cairo': (30.0444, 31.2357),
    'Rio de Janeiro': (-22.9068, -43.1729),
    'Moscow': (55.7558, 37.6173),
    'Los Angeles': (34.0522, -118.2437),
    'Paris': (48.8566, 2.3522),
    'New Delhi': (28.6139, 77.2090)
}

def add_simulated_locations_to_data(df):
    """
    Adds simulated location data to a DataFrame.
    This is for demonstration purposes as the source CSV has no location column.
    """
    locations = list(SAMPLE_LOCATIONS.keys())
    df['location_name'] = [random.choice(locations) for _ in range(len(df))]
    
    # Map the location name to its coordinates
    df['latitude'] = df['location_name'].apply(lambda x: SAMPLE_LOCATIONS[x][0])
    df['longitude'] = df['location_name'].apply(lambda x: SAMPLE_LOCATIONS[x][1])
    
    return df
