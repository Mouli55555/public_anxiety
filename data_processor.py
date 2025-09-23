# data_processor.py
# Handles loading and processing the raw twitter data.

import pandas as pd
import os
import config

def load_data(filepath):
    """
    Loads the tweet data from the specified CSV file.
    Returns a pandas DataFrame or None if the file is not found.
    """
    if not os.path.exists(filepath):
        print(f"Error: The file '{filepath}' was not found.")
        return None
    try:
        # We only need the 'tweet' column for this analysis.
        df = pd.read_csv(filepath, usecols=['tweet'])
        # Drop rows where the tweet is missing to avoid errors.
        df.dropna(subset=['tweet'], inplace=True)
        return df
    except Exception as e:
        print(f"Error loading or reading the CSV file: {e}")
        return None

def analyze_community(community_df, keywords):
    """
    Analyzes a single community's tweets for keyword frequency and score.

    Args:
        community_df (pd.DataFrame): The DataFrame for a specific community.
        keywords (list): A list of keywords to search for.

    Returns:
        pd.DataFrame: A DataFrame with topics, counts, and scores.
    """
    topic_counts = {topic: 0 for topic in keywords}
    total_keyword_mentions = 0

    # Ensure all tweets are strings before processing
    community_df['tweet'] = community_df['tweet'].astype(str)

    for topic in keywords:
        # Use vectorized string operations for efficiency
        count = community_df['tweet'].str.contains(topic, case=False, na=False).sum()
        topic_counts[topic] = count
        total_keyword_mentions += count

    # Calculate the score for each topic
    analysis_data = []
    for topic, count in topic_counts.items():
        score = count / total_keyword_mentions if total_keyword_mentions > 0 else 0
        analysis_data.append({'topic': topic, 'count': count, 'score': score})

    return pd.DataFrame(analysis_data)

def process_and_save_all_communities():
    """
    The main function to orchestrate the data processing.
    It loads the data, splits it into communities, analyzes each,
    and saves the results to CSV files.
    """
    df = load_data(config.INPUT_FILENAME)
    if df is None:
        return False

    # Create the output directory if it doesn't exist
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # Determine the size of each community chunk
    chunk_size = len(df) // config.COMMUNITY_COUNT
    if chunk_size == 0:
        print("Error: Dataset is too small to be split into communities.")
        return False

    # Get the list of keywords from the config file
    keywords_to_analyze = list(config.KEYWORD_WEIGHTS.keys())

    for i in range(config.COMMUNITY_COUNT):
        start_index = i * chunk_size
        end_index = (i + 1) * chunk_size
        community_df_subset = df.iloc[start_index:end_index]

        # Analyze the current community
        analysis_result_df = analyze_community(community_df_subset, keywords_to_analyze)

        # Save the result to a uniquely named CSV file
        output_filepath = os.path.join(config.OUTPUT_DIR, f"community_{i+1}_analysis.csv")
        analysis_result_df.to_csv(output_filepath, index=False)
        print(f"Successfully processed and saved analysis for Community {i+1} to {output_filepath}")

    return True

