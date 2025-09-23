Evaluating Public Anxiety - Refactored Project
This project is a desktop application designed to analyze public anxiety levels by examining keyword frequencies in a dataset of tweets.

This is a refactored version of the original research project, structured for clarity, efficiency, and maintainability.

Team Members (Original Project)

[]

Features
Data Processing: Analyzes a large CSV of tweets and segments them into communities.

Keyword Analysis: Calculates the frequency and score of predefined keywords (e.g., 'anxiety', 'pain', 'love') for each community.

Data Visualization: Generates and displays pie charts to visualize the topic distribution within each community.

Graphical User Interface (GUI): A user-friendly interface built with Tkinter to control the analysis and view results.

How to Run the Application
1. Prerequisites
Python 3.6 or newer.

The required dataset file twitter_English.csv must be present in the same directory as the application files.

2. Install Dependencies
Install the necessary Python libraries using pip and the requirements.txt file. Open your terminal or command prompt in the project directory and run:

pip install -r requirements.txt

3. Run the Application
Execute the main application file from your terminal:

python main_app.py

4. Using the Application
Process Data: When the application launches, first click the "1. Process Tweet Data" button. This will read twitter_English.csv, perform the analysis, and save the results for each community into a new folder named analysis_results.

View Analysis: After the processing is complete, click on any of the "Community" buttons on the left to view a pie chart visualizing the keyword distribution for that specific community.

Project Structure
The refactored code is organized into several key modules:

main_app.py: The main entry point that launches the Tkinter GUI.

config.py: A configuration file to easily manage settings like keywords, file paths, and community size.

data_processor.py: Contains all the logic for loading, processing, and analyzing the tweet data using the pandas library.

visualizer.py: Responsible for generating and embedding matplotlib plots within the Tkinter application.

requirements.txt: Lists all the project dependencies.

analysis_results/: This directory is created automatically to store the CSV output from the data processing step.