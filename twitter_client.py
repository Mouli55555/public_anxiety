# twitter_client.py
# This module is responsible for fetching live data from the Twitter API.

import requests
import pandas as pd
import json

# IMPORTANT: You must get your own Bearer Token from the Twitter Developer Portal.
# Replace the placeholder text below with your actual token.
BEARER_TOKEN = "AAAAAAAAAAAAAAAAAAAAAPBi4QEAAAAAMyYjf9Twue7c5iinGoigvEw46H8%3DCEFcliZSivG8obqnMQQPF4wiZKpnS3QZHTnFg5TeokjW8OLj7J"

SEARCH_URL = "https://api.twitter.com/2/tweets/search/recent"

def fetch_recent_tweets(query, max_results=50):
    """
    Fetches recent tweets from the Twitter API.
    Returns a tuple: (DataFrame, error_message).
    If successful, error_message will be None.
    If fails, DataFrame will be None.
    """
    if not BEARER_TOKEN or BEARER_TOKEN == "YOUR_BEARER_TOKEN_HERE":
        return None, "Twitter Bearer Token is not set in twitter_client.py. Please add your token."

    headers = {
        "Authorization": f"Bearer {BEARER_TOKEN}"
    }
    params = {
        'query': f"{query} lang:en -is:retweet",
        'max_results': max_results,
        'tweet.fields': 'author_id,created_at,text',
        'user.fields': 'name,username' # Request user details
    }

    try:
        response = requests.get(SEARCH_URL, headers=headers, params=params)
        
        # Check for HTTP errors (like 401 Unauthorized, 403 Forbidden)
        if response.status_code != 200:
            try:
                # Try to parse the detailed error message from Twitter
                error_details = response.json()
                error_title = error_details.get('title', 'Unknown Error')
                error_detail_msg = error_details.get('detail', response.text)
                return None, f"API Error ({response.status_code} {error_title}): {error_detail_msg}"
            except json.JSONDecodeError:
                return None, f"API Error (Status {response.status_code}): {response.text}"

        json_response = response.json()
        
        if 'data' not in json_response or not json_response['data']:
            return pd.DataFrame(columns=['Name', 'Email', 'tweet']), None # Return empty DataFrame for no results

        # Process the JSON response into a clean DataFrame
        tweets_data = []
        for tweet in json_response['data']:
            tweets_data.append({
                'Name': f"User_{tweet['author_id']}",
                'Email': f"user_{tweet['author_id']}@twitter.com", # Email is not provided by API
                'tweet': tweet['text']
            })
            
        return pd.DataFrame(tweets_data), None

    except requests.exceptions.RequestException as e:
        return None, f"A network error occurred. Please check your internet connection. Details: {e}"
    except Exception as e:
        return None, f"An unexpected error occurred: {e}"

