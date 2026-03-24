# config.py
# Central configuration for the project.

import os
from pathlib import Path

from dotenv import load_dotenv

try:
    import streamlit as st
except Exception:
    st = None


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env", override=False)


def _read_streamlit_secret(*names):
    if st is None:
        return ""

    try:
        for name in names:
            if name in st.secrets:
                value = str(st.secrets[name]).strip()
                if value:
                    return value
    except Exception:
        return ""

    return ""


def _read_setting(*names, default=""):
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value

    secret_value = _read_streamlit_secret(*names)
    if secret_value:
        return secret_value

    return default

# --- File Paths ---
# Use absolute repo paths so Cloud deployment does not depend on the process CWD.
INPUT_FILENAME = BASE_DIR / "twitter_English.csv"

# The directory where the processed community analysis CSV files will be saved.
OUTPUT_DIR = BASE_DIR / "community_analysis_results"

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

# --- Live Integrations ---
# Values are loaded from environment variables first. Because `.env` is loaded
# into the environment above, local development still works without extra setup.
# Streamlit secrets act as a fallback on Community Cloud.
GEMINI_API_KEY = _read_setting("GEMINI_API_KEY", "GOOGLE_API_KEY")
GEMINI_MODEL = _read_setting("GEMINI_MODEL", default="gemini-2.5-flash")
BLUESKY_API_BASE_URL = _read_setting("BLUESKY_API_BASE_URL", default="https://api.bsky.app")
