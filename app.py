# app.py
# This is the main application file for the Streamlit-based UI.

import streamlit as st
import pandas as pd
import os
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium

# Import the backend logic from our other modules
import config
import data_processor
from predictor import FuzzyAnxietyPredictor
import twitter_client
import location_extractor

# --- Page Configuration ---
st.set_page_config(
    page_title="Public Anxiety Analysis Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- App Styling ---
st.markdown("""
<style>
    .reportview-container { background: #f0f2f6; }
    .sidebar .sidebar-content { background: #ffffff; }
    .stButton>button {
        color: #ffffff; background-color: #0068c9; border-radius: 8px;
        border: none; padding: 10px 20px;
    }
    .stTextInput>div>div>input, .stTextArea>div>textarea { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# --- Session State Initialization ---
if 'keywords' not in st.session_state:
    st.session_state.keywords = config.KEYWORD_WEIGHTS.copy()
if 'fuzzy_predictor' not in st.session_state:
    st.session_state.fuzzy_predictor = FuzzyAnxietyPredictor(st.session_state.keywords)
# Add a state for our map to prevent it from disappearing
if 'anxiety_map' not in st.session_state:
    st.session_state.anxiety_map = None


# --- Helper function to re-initialize predictor ---
def update_predictor():
    st.session_state.fuzzy_predictor = FuzzyAnxietyPredictor(st.session_state.keywords)
    st.success("Fuzzy predictor updated with new keywords!")

# --- Sidebar ---
with st.sidebar:
    st.header("Controls & Analysis")
    st.markdown("Use the options below to run the analysis.")

    if st.button("Step 1: Process Community Data"):
        with st.spinner("Processing all communities..."):
            success = data_processor.process_and_save_all_communities()
            if success:
                st.success("Community data processed!")
                st.balloons()
            else:
                st.error("Processing failed. Check source file.")
    st.markdown("---")

    st.header("Keyword Management")
    st.markdown("Modify keyword weights and update the fuzzy system.")
    
    for keyword, weight in list(st.session_state.keywords.items()):
        col1, col2, col3 = st.columns([3, 2, 1])
        new_weight = col1.number_input(f"Weight for '{keyword}'", value=weight, key=f"weight_{keyword}", step=0.1, format="%.1f")
        st.session_state.keywords[keyword] = new_weight
        if col3.button("X", key=f"del_{keyword}"):
            del st.session_state.keywords[keyword]
            st.rerun()

    st.markdown("##### Add New Keyword")
    col1, col2 = st.columns([3, 2])
    new_keyword_name = col1.text_input("Keyword", key="new_keyword_name")
    new_keyword_weight = col2.number_input("Weight", value=1.0, key="new_keyword_weight", step=0.1)
    if st.button("Add Keyword"):
        if new_keyword_name and new_keyword_name not in st.session_state.keywords:
            st.session_state.keywords[new_keyword_name] = new_keyword_weight
            st.rerun()
        else:
            st.warning("Keyword is empty or already exists.")
            
    if st.button("Update Fuzzy Predictor with Changes", on_click=update_predictor):
        pass

# --- Main Application UI ---
st.title("🧠 Estimating Public Anxiety from Social Media")
st.markdown("A dashboard for analyzing tweet data using statistical, fuzzy logic, and live data approaches.")
st.markdown("---")

# --- Tab Layout ---
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Community Analysis", 
    "Fuzzy Anxiety Prediction", 
    "Historical Trend Analysis", 
    "High-Risk Identification",
    "Geospatial Map",
    "Live Twitter Analysis"
])

with tab1:
    st.header("Community Keyword Analysis")
    st.markdown("Select a community to visualize the distribution of keywords.")
    output_files = [f for f in os.listdir(config.OUTPUT_DIR) if f.endswith('.csv')] if os.path.exists(config.OUTPUT_DIR) else []
    if not output_files:
        st.info("Please process the community data first using the button in the sidebar.")
    else:
        community_files = sorted([os.path.join(config.OUTPUT_DIR, f) for f in output_files])
        selected_file = st.selectbox(
            "Choose a community analysis file:", community_files,
            format_func=lambda x: os.path.basename(x).replace('_', ' ').replace('.csv', '').title()
        )
        if selected_file:
            df = pd.read_csv(selected_file)
            df_display = df[df['count'] > 0]
            if not df_display.empty:
                fig = px.pie(df_display, values='count', names='topic', title=f"Keyword Distribution for {os.path.basename(selected_file).replace('_', ' ').replace('.csv', '').title()}")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No keywords found for this community.")

with tab2:
    st.header("Fuzzy Logic Anxiety Prediction")
    st.markdown("Enter a tweet to analyze its anxiety level using the updated Fuzzy Inference System.")
    user_tweet_input = st.text_area("Enter tweet text:", "I'm feeling so stressed and anxious about the upcoming exams, the pressure is just overwhelming.", height=100)
    if st.button("Analyze Tweet Anxiety"):
        if user_tweet_input:
            with st.spinner("Running fuzzy analysis..."):
                predictor = st.session_state.fuzzy_predictor
                anxiety_score = predictor.compute_prediction(user_tweet_input)
                anxiety_level = predictor.interpret_anxiety_score(anxiety_score)

                col1, col2 = st.columns(2)
                col1.metric("Predicted Anxiety Score", f"{anxiety_score:.2f} / 10")
                if "High" in anxiety_level: col2.error(f"Interpreted Level: **{anxiety_level}**")
                elif "Moderate" in anxiety_level: col2.warning(f"Interpreted Level: **{anxiety_level}**")
                else: col2.success(f"Interpreted Level: **{anxiety_level}**")
                
                with st.expander("Show Analysis Breakdown"):
                    st.markdown(f"- **Sentiment Score:** `{predictor.get_sentiment_score(user_tweet_input):.2f}`")
                    st.markdown(f"- **Weighted Keyword Score:** `{predictor.get_keyword_score(user_tweet_input):.2f}`")

with tab3:
    st.header("Historical Trend Analysis")
    st.markdown("Analyze the anxiety trend across the entire dataset over time.")
    chunk_size = st.slider("Select analysis window size (number of tweets):", 50, 500, 100, 50)
    if st.button("Run Trend Analysis"):
        df = data_processor.load_data(config.INPUT_FILENAME)
        if df is not None and not df.empty:
            with st.spinner(f"Analyzing trends with a window size of {chunk_size}..."):
                predictor = st.session_state.fuzzy_predictor
                trend_scores = []
                for i in range(0, len(df), chunk_size):
                    chunk = df.iloc[i:i + chunk_size]
                    avg_score = chunk['tweet'].apply(predictor.compute_prediction).mean()
                    trend_scores.append(avg_score)
                
                trend_df = pd.DataFrame({'Time Interval': range(len(trend_scores)), 'Average Anxiety Score': trend_scores})
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=trend_df['Time Interval'], y=trend_df['Average Anxiety Score'], mode='lines+markers', name='Anxiety Trend'))
                fig.update_layout(title='Public Anxiety Trend Over Time', xaxis_title='Time Interval (Batch of Tweets)', yaxis_title='Average Anxiety Score')
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Could not load data to perform trend analysis.")
            
with tab4:
    st.header("High-Risk Individual Identification")
    st.markdown("Scan the dataset to find individuals whose tweets show a high level of anxiety.")
    anxiety_threshold = st.slider("Anxiety Score Threshold:", min_value=0.0, max_value=10.0, value=7.5, step=0.1)
    if st.button("Find High-Anxiety Individuals"):
        full_df = data_processor.load_full_data(config.INPUT_FILENAME)
        if full_df is not None and not full_df.empty:
            with st.spinner("Analyzing all tweets... This might take time."):
                predictor = st.session_state.fuzzy_predictor
                full_df['anxiety_score'] = full_df['tweet'].apply(predictor.compute_prediction)
                high_risk_df = full_df[full_df['anxiety_score'] >= anxiety_threshold].copy()
                high_risk_df.sort_values(by='anxiety_score', ascending=False, inplace=True)
            st.success(f"Found **{len(high_risk_df)}** individuals matching the criteria.")
            if not high_risk_df.empty:
                st.dataframe(high_risk_df[['Name', 'Email', 'tweet', 'anxiety_score']], use_container_width=True)
        else:
            st.error("Could not load the full dataset.")

with tab5:
    st.header("Geospatial Anxiety Hotspot Map")
    st.markdown("Visualize the geographic distribution of high-anxiety tweets from the dataset.")
    
    map_threshold = st.slider("Anxiety Score Threshold for Mapping:", min_value=0.0, max_value=10.0, value=8.0, step=0.1, key="map_slider")
    
    if st.button("Generate Anxiety Map"):
        full_df = data_processor.load_full_data(config.INPUT_FILENAME)
        if full_df is not None and not full_df.empty:
            with st.spinner("Analyzing and mapping data..."):
                full_df = location_extractor.add_simulated_locations_to_data(full_df)
                predictor = st.session_state.fuzzy_predictor
                full_df['anxiety_score'] = full_df['tweet'].apply(predictor.compute_prediction)
                hotspot_df = full_df[full_df['anxiety_score'] >= map_threshold]
            
            st.success(f"Found {len(hotspot_df)} high-anxiety tweets to map.")
            
            if not hotspot_df.empty:
                map_center = [hotspot_df['latitude'].mean(), hotspot_df['longitude'].mean()]
                anxiety_map = folium.Map(location=map_center, zoom_start=2)
                for idx, row in hotspot_df.iterrows():
                    popup_text = f"<b>Score: {row['anxiety_score']:.2f}</b><br><i>{row['tweet'][:100]}...</i>"
                    folium.CircleMarker(
                        location=[row['latitude'], row['longitude']], radius=5,
                        popup=popup_text, color='red', fill=True, fill_color='darkred'
                    ).add_to(anxiety_map)
                # ** THE FIX: Store the generated map in the session state **
                st.session_state.anxiety_map = anxiety_map
            else:
                # If no data, clear any previous map
                st.session_state.anxiety_map = None
                st.warning("No tweets found above the threshold to map.")
        else:
            st.error("Could not load data to generate the map.")

    # ** THE FIX: Display the map from session state, if it exists **
    if st.session_state.anxiety_map:
        st_folium(st.session_state.anxiety_map, width=725, height=500)


with tab6:
    st.header("Live Twitter Anxiety Analysis")
    st.markdown("Fetch and analyze recent tweets directly from the X (Twitter) API.")
    live_query = st.text_input("Enter a search query:", "anxiety OR depression OR lonely")
    num_tweets = st.slider("Number of tweets to fetch:", 10, 100, 50)
    if st.button("Fetch & Analyze Live Tweets"):
        with st.spinner("Fetching live tweets..."):
            live_df, error_message = twitter_client.fetch_recent_tweets(live_query, max_results=num_tweets)
        if error_message:
            st.error(f"**Failed to fetch tweets.**\n\nDetails: {error_message}")
        elif live_df.empty:
            st.warning("No recent tweets found for your query.")
        else:
            st.success(f"Successfully fetched {len(live_df)} tweets.")
            with st.spinner("Analyzing live tweets..."):
                predictor = st.session_state.fuzzy_predictor
                live_df['anxiety_score'] = live_df['tweet'].apply(predictor.compute_prediction)
                live_df.sort_values(by='anxiety_score', ascending=False, inplace=True)
            st.dataframe(live_df[['Name', 'tweet', 'anxiety_score']], use_container_width=True)

