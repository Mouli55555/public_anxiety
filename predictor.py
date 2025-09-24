# predictor.py
# This module contains the logic for the Fuzzy Inference System.

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk

# --- One-time setup for VADER ---
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    print("Downloading VADER lexicon for sentiment analysis...")
    nltk.download('vader_lexicon')

class FuzzyAnxietyPredictor:
    def __init__(self, keyword_weights):
        """
        Initializes the Fuzzy Anxiety Predictor system.
        This can be re-initialized with new keyword weights.
        """
        self.analyzer = SentimentIntensityAnalyzer()
        self.keyword_weights = keyword_weights
        print(f"Initializing predictor with {len(self.keyword_weights)} keywords.")
        self._setup_fuzzy_system()
        self.anxiety_simulation = ctrl.ControlSystemSimulation(self.anxiety_ctrl)

    def _setup_fuzzy_system(self):
        """
        Defines the antecedents, consequent, and rules for the fuzzy system.
        """
        # Antecedents, Consequent, Membership Functions, and Rules...
        # (This logic remains the same as before)
        sentiment = ctrl.Antecedent(np.arange(-1, 1.01, 0.1), 'sentiment')
        keyword_score = ctrl.Antecedent(np.arange(0, 10.01, 0.5), 'keyword_score')
        anxiety_level = ctrl.Consequent(np.arange(0, 10.01, 0.5), 'anxiety_level')

        sentiment['negative'] = fuzz.trimf(sentiment.universe, [-1, -0.5, 0])
        sentiment['neutral'] = fuzz.trimf(sentiment.universe, [-0.2, 0, 0.2])
        sentiment['positive'] = fuzz.trimf(sentiment.universe, [0, 0.5, 1])

        keyword_score['low'] = fuzz.trimf(keyword_score.universe, [0, 0, 4])
        keyword_score['medium'] = fuzz.trimf(keyword_score.universe, [2, 5, 8])
        keyword_score['high'] = fuzz.trimf(keyword_score.universe, [6, 10, 10])

        anxiety_level['low'] = fuzz.trimf(anxiety_level.universe, [0, 0, 4])
        anxiety_level['moderate'] = fuzz.trimf(anxiety_level.universe, [2, 5, 8])
        anxiety_level['high'] = fuzz.trimf(anxiety_level.universe, [6, 10, 10])

        rule1 = ctrl.Rule(sentiment['negative'] | keyword_score['high'], anxiety_level['high'])
        rule2 = ctrl.Rule(sentiment['neutral'] & keyword_score['medium'], anxiety_level['moderate'])
        rule3 = ctrl.Rule(sentiment['positive'] & keyword_score['low'], anxiety_level['low'])
        rule4 = ctrl.Rule(sentiment['negative'] & keyword_score['medium'], anxiety_level['high'])

        self.anxiety_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4])


    def get_sentiment_score(self, text):
        return self.analyzer.polarity_scores(text)['compound']

    def get_keyword_score(self, text):
        score = 0
        text_lower = text.lower()
        if not self.keyword_weights: return 0
        
        for keyword, weight in self.keyword_weights.items():
            if keyword in text_lower:
                score += weight
        
        # Normalize score based on the current max possible score
        max_score = sum(w for w in self.keyword_weights.values() if w > 0)
        return (score / max_score) * 10 if max_score > 0 else 0

    def compute_prediction(self, text):
        # Ensure text is a string
        if not isinstance(text, str):
            return 0.0

        sentiment_val = self.get_sentiment_score(text)
        keyword_val = self.get_keyword_score(text)

        self.anxiety_simulation.input['sentiment'] = sentiment_val
        self.anxiety_simulation.input['keyword_score'] = keyword_val
        self.anxiety_simulation.compute()

        return self.anxiety_simulation.output['anxiety_level']

    def interpret_anxiety_score(self, score):
        if score > 7.5: return "Very High Anxiety"
        elif score > 5.5: return "High Anxiety"
        elif score > 3.5: return "Moderate Anxiety"
        else: return "Low Anxiety"

# Note: We no longer create a default instance here.
# The instance is created and managed in the app.py file.

