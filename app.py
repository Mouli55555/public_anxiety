# app.py
# This is the main application file for the Streamlit-based UI.

import streamlit as st
import pandas as pd
import os
import plotly.express as px

# Import the backend logic from our other modules
import config
import data_processor
from predictor import fuzzy_predictor

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
    .reportview-container {
        background: #f0f2f6;
    }
    .sidebar .sidebar-content {
        background: #ffffff;
    }
    .stButton>button {
        color: #ffffff;
        background-color: #0068c9;
        border-radius: 8px;
        border: none;
        padding: 10px 20px;
    }
    .stTextInput>div>div>input {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


# --- Main Application UI ---

st.title("🧠 Estimating Public Anxiety from Social Media")
st.markdown("A dashboard for analyzing tweet data using statistical and fuzzy logic-based approaches.")
st.markdown("---")


# --- Sidebar for Controls ---
st.sidebar.header("Controls & Analysis")
st.sidebar.markdown("Use the options below to run the analysis.")

# Button to run the initial data processing
if st.sidebar.button("Step 1: Process Community Data"):
    with st.spinner("Processing all communities... this may take a moment."):
        success = data_processor.process_and_save_all_communities()
        if success:
            st.sidebar.success("Community data processed successfully!")
            st.balloons()
        else:
            st.sidebar.error("Failed to process data. Check the source file.")

st.sidebar.markdown("---")


# --- Main Content Area ---

# Section 1: Community Analysis Visualization
st.header("Community Keyword Analysis")
st.markdown("Select a community to visualize the distribution of keywords.")

# Dropdown to select a community
output_files = [f for f in os.listdir(config.OUTPUT_DIR) if f.endswith('.csv')] if os.path.exists(config.OUTPUT_DIR) else []
if not output_files:
    st.info("Please process the community data first using the button in the sidebar.")
else:
    community_files = sorted([os.path.join(config.OUTPUT_DIR, f) for f in output_files])
    selected_community_file = st.selectbox(
        "Choose a community analysis file:",
        community_files,
        format_func=lambda x: os.path.basename(x).replace('_', ' ').replace('.csv', '').title()
    )

    if selected_community_file:
        try:
            df = pd.read_csv(selected_community_file)
            df_display = df[df['count'] > 0]

            if not df_display.empty:
                fig = px.pie(
                    df_display,
                    values='count',
                    names='topic',
                    title=f"Keyword Distribution for {os.path.basename(selected_community_file).replace('_', ' ').replace('.csv', '').title()}",
                    hover_data=['score'],
                    labels={'topic': 'Keyword', 'count': 'Frequency'}
                )
                fig.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No keywords found for this community.")

        except Exception as e:
            st.error(f"Could not load or display the chart. Error: {e}")


st.markdown("---")

# Section 2: Fuzzy Logic Anxiety Prediction
st.header("Fuzzy Logic Anxiety Prediction")
st.markdown("Enter a tweet below to analyze its anxiety level using the Fuzzy Inference System.")

user_tweet_input = st.text_area(
    "Enter tweet text for analysis:",
    "I'm feeling so stressed and anxious about the upcoming exams, the pressure is just overwhelming.",
    height=100
)

if st.button("Analyze Tweet Anxiety"):
    if user_tweet_input:
        with st.spinner("Running fuzzy analysis..."):
            # ================================================================= #
            # THIS IS THE FIX: The method is called `compute_prediction`.       #
            # The old, incorrect name was `predict_anxiety`.                    #
            anxiety_score = fuzzy_predictor.compute_prediction(user_tweet_input)
            # ================================================================= #
            
            anxiety_level = fuzzy_predictor.interpret_anxiety_score(anxiety_score)

            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    label="Predicted Anxiety Score",
                    value=f"{anxiety_score:.2f} / 10"
                )
            with col2:
                if "High" in anxiety_level or "Very High" in anxiety_level:
                    st.error(f"Interpreted Level: **{anxiety_level}**")
                elif "Moderate" in anxiety_level:
                    st.warning(f"Interpreted Level: **{anxiety_level}**")
                else:
                    st.success(f"Interpreted Level: **{anxiety_level}**")

            st.subheader("Analysis Breakdown")
            sentiment = fuzzy_predictor.get_sentiment_score(user_tweet_input)
            keyword_score = fuzzy_predictor.get_keyword_score(user_tweet_input)
            st.markdown(f"- **Sentiment Score:** `{sentiment:.2f}` (from VADER)")
            st.markdown(f"- **Weighted Keyword Score:** `{keyword_score:.2f}` (from weighted keywords)")
    else:
        st.warning("Please enter a tweet to analyze.")

