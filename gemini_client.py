# gemini_client.py
# This module is responsible for interacting with the Google Gemini API.

import config


def _load_genai_client():
    try:
        import google.generativeai as genai
        return genai, None
    except Exception as exc:
        return None, str(exc)


def _build_history(chat_history):
    return [
        {
            "role": msg["role"],
            "parts": [{"text": part} for part in msg["parts"] if part]
        }
        for msg in chat_history if "parts" in msg and msg["parts"]
    ]

def get_gemini_response(api_key, chat_history):
    """
    Gets a response from the Gemini model using the official client library.

    Args:
        api_key (str): Your Google AI API key.
        chat_history (list): A list of conversation history dicts.

    Returns:
        str: The model's response or an error message.
    """
    if not api_key:
        return (
            "Gemini API key is not configured. Add GEMINI_API_KEY to Streamlit "
            "secrets or your environment, then restart the app."
        )

    genai, import_error = _load_genai_client()
    if import_error:
        return f"Gemini support is unavailable right now: {import_error}"

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(config.GEMINI_MODEL)
        response = model.generate_content(_build_history(chat_history))
        response_text = getattr(response, "text", "").strip()
        if response_text:
            return response_text
        return "Sorry, I received an empty response from the AI."
    except Exception as e:
        error_text = str(e)
        if "404" in error_text and "models/" in error_text:
            return (
                f"Gemini model '{config.GEMINI_MODEL}' is not available for generateContent. "
                "Update GEMINI_MODEL in Streamlit secrets or your environment to a "
                "supported model such as 'gemini-2.5-flash', then restart the app."
            )
        return f"An unexpected error occurred: {error_text}"
