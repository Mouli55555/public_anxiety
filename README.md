Evaluating Public Anxiety
This project is a Streamlit dashboard for exploring anxiety-related patterns in a tweet dataset. It includes community keyword analysis, fuzzy-logic scoring, trend analysis, hotspot mapping, an optional live X/Twitter-style feed powered by Bluesky, and an optional Gemini-based support chat tab.

Features
Data Processing: Splits the dataset into communities and saves per-community keyword analysis.

Fuzzy Anxiety Prediction: Scores tweet text with VADER sentiment and a fuzzy inference system.

Trend Analysis: Charts average anxiety across dataset windows.

Risk and Map Views: Highlights high-anxiety tweets and plots simulated locations on a map.

Optional Live Integrations: Supports Gemini and a live X/Twitter-style feed powered by a free public Bluesky search source. Gemini uses API credentials; Bluesky live search does not.

How to Run the Application
1. Prerequisites
Python 3.10 or newer is recommended.

The dataset file `twitter_English.csv` must be present in the project root.

2. Create and activate a virtual environment
On Windows PowerShell:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3. Install dependencies

```powershell
pip install -r requirements.txt
```

4. Optional API configuration
Create a `.env` file in the project root for the optional live API-powered tabs:

```powershell
GEMINI_API_KEY=your-gemini-api-key
GEMINI_MODEL=gemini-2.5-flash
BLUESKY_API_BASE_URL=https://api.bsky.app
```

You can still set real OS environment variables instead. If both exist, the OS environment value wins.

5. Run the Streamlit app

```powershell
streamlit run app.py
```

Deploying to Streamlit Community Cloud
Use `app.py` as the entrypoint.

Add `GEMINI_API_KEY` in the app's Secrets settings if you want the AI coach tab to work.

This repo now reads configuration from Streamlit secrets, environment variables, or `.env`.

Choose the Python version from Streamlit Cloud's "Advanced settings" during deployment. `runtime.txt` is not used on Community Cloud.

Project Structure
`app.py`: Streamlit entry point and dashboard UI.

`config.py`: Central configuration and environment-backed API settings.

`data_processor.py`: CSV loading and community analysis logic.

`predictor.py`: Fuzzy inference system for anxiety scoring.

`twitter_client.py`: Optional live social feed fetch logic, powered by Bluesky.

`gemini_client.py`: Optional Gemini chat integration.

`community_analysis_results/`: Generated CSV outputs from processed community data.
