# config.py
# Central configuration for the project.

import os

# --- File Paths ---
# The name of the input CSV file containing the raw tweet data.
# This file is expected to be in the same directory as the scripts.
INPUT_FILENAME = "twitter_English.csv"

# The directory where the processed community analysis CSV files will be saved.
OUTPUT_DIR = "community_analysis_results"

# --- Analysis Parameters ---
# The number of communities to split the dataset into.
# Each community will have an equal number of tweets.
COMMUNITY_COUNT = 5

# A dictionary of keywords and their assigned weights.
# Higher weights indicate a stronger impact on the anxiety score.
# This is used by the fuzzy logic prediction system.
KEYWORD_WEIGHTS = {
    'anxiety': 2.0,
    'suicide': 5.0,  # Highest weight due to severity
    'lockdown': 1.0,
    'please': 0.5,
    'chronic': 1.5,
    'fatigue': 1.0,
    'love': -1.0,  # Negative weight can counteract anxiety
    'sad': 2.0,
    'pain': 2.5,
    'depression': 3.0,
    'stressed': 2.0,
    'overwhelmed': 1.5
}

