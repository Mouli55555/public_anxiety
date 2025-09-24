# data_processor.py
# Handles loading and processing the raw twitter data.

import pandas as pd
import os
import config

def load_data(filepath):
    """
    Loads only the tweet column for efficient community analysis.
    """
    if not os.path.exists(filepath):
        print(f"Error: The file '{filepath}' was not found.")
        return None
    try:
        df = pd.read_csv(filepath, usecols=['tweet'])
        df.dropna(subset=['tweet'], inplace=True)
        return df
    except Exception as e:
        print(f"Error loading or reading the CSV file: {e}")
        return None

def load_full_data(filepath):
    """
    NEW: Loads all relevant columns (tweet, Name, Email) for high-risk analysis.
    """
    if not os.path.exists(filepath):
        print(f"Error: The file '{filepath}' was not found.")
        return None
    try:
        # Load all columns needed for identifying individuals
        df = pd.read_csv(filepath, usecols=['tweet', 'Name', 'Email'])
        # Drop rows where any of the crucial data is missing
        df.dropna(subset=['tweet', 'Name', 'Email'], inplace=True)
        return df
    except Exception as e:
        print(f"Error loading or reading the full CSV file: {e}")
        return None

def analyze_community(community_df, keywords):
    """
    Analyzes a single community's tweets for keyword frequency and score.
    """
    topic_counts = {topic: 0 for topic in keywords}
    total_keyword_mentions = 0

    community_df['tweet'] = community_df['tweet'].astype(str)

    for topic in keywords:
        count = community_df['tweet'].str.contains(topic, case=False, na=False).sum()
        topic_counts[topic] = count
        total_keyword_mentions += count

    analysis_data = []
    for topic, count in topic_counts.items():
        score = count / total_keyword_mentions if total_keyword_mentions > 0 else 0
        analysis_data.append({'topic': topic, 'count': count, 'score': score})

    return pd.DataFrame(analysis_data)

def process_and_save_all_communities():
    """
    The main function to orchestrate the data processing.
    """
    df = load_data(config.INPUT_FILENAME)
    if df is None:
        return False

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    chunk_size = len(df) // config.COMMUNITY_COUNT
    if chunk_size == 0:
        print("Error: Dataset is too small to be split into communities.")
        return False

    keywords_to_analyze = list(config.KEYWORD_WEIGHTS.keys())

    for i in range(config.COMMUNITY_COUNT):
        start_index = i * chunk_size
        end_index = (i + 1) * chunk_size
        community_df_subset = df.iloc[start_index:end_index]
        analysis_result_df = analyze_community(community_df_subset, keywords_to_analyze)
        output_filepath = os.path.join(config.OUTPUT_DIR, f"community_{i+1}_analysis.csv")
        analysis_result_df.to_csv(output_filepath, index=False)
        print(f"Successfully processed and saved analysis for Community {i+1}")

    return True

