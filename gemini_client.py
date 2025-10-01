# gemini_client.py
# This module is responsible for interacting with the Google Gemini API using the 'requests' library.

import requests
import json

# **THE FIX:** Use a model name that is available on your specific account list.
API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro-latest:generateContent"

def get_gemini_response(api_key, chat_history):
    """
    Gets a response from the Gemini model using a direct REST API call.

    Args:
        api_key (str): Your Google AI API key.
        chat_history (list): A list of conversation history dicts.

    Returns:
        str: The model's response or an error message.
    """
    headers = {
        "Content-Type": "application/json",
        "X-goog-api-key": api_key  # Send the API key as a header
    }
    
    # The REST API expects a specific JSON format. We convert our chat history to it.
    valid_history = [
        {
            "role": msg["role"],
            "parts": [{"text": part} for part in msg["parts"] if part]
        }
        for msg in chat_history if "parts" in msg and msg["parts"]
    ]

    payload = {
        "contents": valid_history
    }
    
    try:
        response = requests.post(API_URL, headers=headers, data=json.dumps(payload))
        
        # Raise an exception if the HTTP request returned an error
        response.raise_for_status()
        
        response_json = response.json()
        
        # Extract the text from the response
        if "candidates" in response_json and response_json["candidates"]:
            first_candidate = response_json["candidates"][0]
            if "content" in first_candidate and "parts" in first_candidate["content"]:
                return first_candidate["content"]["parts"][0]["text"]
        
        return "Sorry, I received an empty response from the AI."

    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
        return f"An error occurred connecting to the service (HTTP {response.status_code}). Details: {response.text}"
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return f"An unexpected error occurred: {e}"