# predictor.py
# Advanced hybrid fuzzy anxiety predictor with dataset-calibrated signals.

import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# vaderSentiment bundles its own lexicon, so we do not need NLTK downloads.

STOPWORDS = {
    "a", "about", "after", "all", "also", "am", "an", "and", "any", "are", "as",
    "at", "be", "because", "been", "being", "but", "by", "can", "could", "did",
    "do", "does", "doing", "for", "from", "get", "got", "had", "has", "have",
    "he", "her", "here", "hers", "him", "his", "how", "i", "if", "in", "into",
    "is", "it", "its", "just", "like", "me", "more", "most", "my", "no", "not",
    "now", "of", "on", "only", "or", "our", "out", "over", "she", "so", "some",
    "that", "the", "their", "them", "there", "they", "this", "to", "too", "up",
    "us", "was", "we", "were", "what", "when", "which", "who", "why", "will",
    "with", "you", "your",
}

NEGATORS = {
    "aint", "aren't", "cannot", "cant", "can't", "didn't", "didnt", "doesn't",
    "doesnt", "don't", "dont", "hardly", "isn't", "isnt", "never", "no", "none",
    "not", "nothing", "rarely", "wasn't", "wasnt", "without", "won't", "wont",
}

INTENSIFIERS = {
    "absolutely", "extremely", "highly", "incredibly", "really", "so", "super",
    "too", "totally", "very",
}

FIRST_PERSON = {"i", "me", "my", "mine", "myself"}

DISTRESS_TERMS = {
    "anxiety": 3.4,
    "anxious": 3.4,
    "panic": 3.6,
    "panicking": 3.8,
    "stressed": 2.6,
    "stress": 2.4,
    "overwhelmed": 2.9,
    "overthinking": 2.6,
    "worried": 1.9,
    "nervous": 1.8,
    "fear": 2.0,
    "scared": 1.9,
    "terrified": 3.0,
    "depression": 3.7,
    "depressed": 3.7,
    "hopeless": 3.9,
    "worthless": 4.2,
    "lonely": 2.4,
    "alone": 2.0,
    "isolated": 2.8,
    "isolation": 2.8,
    "fatigue": 1.7,
    "exhausted": 2.2,
    "burnout": 2.8,
    "meltdown": 3.5,
    "crying": 2.2,
    "cry": 1.6,
    "pressure": 1.7,
    "insomnia": 3.2,
    "sleepless": 2.8,
    "trauma": 3.2,
    "ptsd": 3.8,
    "suicide": 5.0,
    "suicidal": 5.0,
    "chronic": 1.4,
    "pain": 2.1,
}

DISTRESS_PHRASES = {
    "panic attack": 4.5,
    "panic attacks": 4.5,
    "social anxiety": 3.8,
    "mental breakdown": 4.0,
    "breaking down": 3.8,
    "falling apart": 3.8,
    "cant sleep": 3.6,
    "can't sleep": 3.6,
    "cannot sleep": 3.6,
    "cant cope": 3.8,
    "can't cope": 3.8,
    "cannot cope": 3.8,
    "cant breathe": 4.4,
    "can't breathe": 4.4,
    "cannot breathe": 4.4,
    "hard to breathe": 3.8,
    "heart racing": 3.2,
    "feel empty": 2.8,
    "feel trapped": 3.3,
    "so stressed": 2.8,
    "so anxious": 3.6,
    "need help": 2.2,
    "emotional pain": 3.4,
}

PROTECTIVE_TERMS = {
    "better": 1.8,
    "calm": 2.0,
    "calmer": 2.0,
    "grateful": 2.1,
    "happy": 1.5,
    "healing": 2.4,
    "hopeful": 2.5,
    "improving": 1.8,
    "okay": 1.0,
    "peaceful": 2.1,
    "recovering": 2.3,
    "relaxed": 1.8,
    "safe": 1.5,
    "stable": 1.6,
    "support": 1.4,
    "supported": 1.4,
    "therapy": 1.1,
    "thankful": 1.9,
    "love": 0.8,
}

PROTECTIVE_PHRASES = {
    "doing better": 2.4,
    "feel better": 2.2,
    "feeling better": 2.2,
    "getting better": 2.4,
    "i am okay": 1.7,
    "i'm okay": 1.7,
    "much better": 2.4,
    "on the mend": 2.6,
    "support system": 2.2,
    "taking a break": 1.5,
    "thank you": 1.4,
}

CRISIS_PHRASES = {
    "end my life": 6.0,
    "kill myself": 6.0,
    "self harm": 6.0,
    "suicidal thoughts": 6.0,
    "want to die": 6.0,
}


class FuzzyAnxietyPredictor:
    _learned_weight_cache = {}

    def __init__(self, keyword_weights):
        """
        Initializes the advanced anxiety predictor.
        The final score still comes from a fuzzy system, but now it is fed by:
        - better sentiment features
        - keyword and phrase matching with modifiers
        - dataset-calibrated lexical signals
        - intensity / rumination cues
        - protective cues that reduce false positives
        """
        self.analyzer = SentimentIntensityAnalyzer()
        self.keyword_weights = keyword_weights or {}
        self.training_data_path = Path(__file__).resolve().parent / "twitter_English.csv"
        cache_key = str(self.training_data_path.resolve())
        if cache_key not in self._learned_weight_cache:
            self._learned_weight_cache[cache_key] = self._fit_dataset_signal(self.training_data_path)
        self.learned_signal_weights = dict(self._learned_weight_cache[cache_key])
        self.learned_positive_terms = {
            term: weight for term, weight in self.learned_signal_weights.items() if weight > 0
        }
        self.learned_negative_terms = {
            term: abs(weight) for term, weight in self.learned_signal_weights.items() if weight < 0
        }
        print(
            "Initializing predictor with "
            f"{len(self.keyword_weights)} custom keywords and "
            f"{len(self.learned_signal_weights)} learned lexical features."
        )
        self._setup_fuzzy_system()

    def _setup_fuzzy_system(self):
        """
        Defines a deeper fuzzy system with richer features.
        """
        negative_sentiment = ctrl.Antecedent(np.arange(0, 10.1, 0.5), "negative_sentiment")
        distress_signal = ctrl.Antecedent(np.arange(0, 10.1, 0.5), "distress_signal")
        learned_signal = ctrl.Antecedent(np.arange(0, 10.1, 0.5), "learned_signal")
        intensity_signal = ctrl.Antecedent(np.arange(0, 10.1, 0.5), "intensity_signal")
        protective_signal = ctrl.Antecedent(np.arange(0, 10.1, 0.5), "protective_signal")
        anxiety_level = ctrl.Consequent(np.arange(0, 10.1, 0.5), "anxiety_level")

        for antecedent in (
            negative_sentiment,
            distress_signal,
            learned_signal,
            intensity_signal,
            protective_signal,
        ):
            antecedent["low"] = fuzz.trapmf(antecedent.universe, [0, 0, 2, 4])
            antecedent["medium"] = fuzz.trimf(antecedent.universe, [2.5, 5, 7.5])
            antecedent["high"] = fuzz.trapmf(antecedent.universe, [6, 8, 10, 10])

        anxiety_level["low"] = fuzz.trapmf(anxiety_level.universe, [0, 0, 2, 4])
        anxiety_level["moderate"] = fuzz.trimf(anxiety_level.universe, [2.5, 4.8, 6.5])
        anxiety_level["high"] = fuzz.trimf(anxiety_level.universe, [5.5, 7.4, 9.0])
        anxiety_level["severe"] = fuzz.trapmf(anxiety_level.universe, [8.0, 9.0, 10.0, 10.0])

        rules = [
            ctrl.Rule(distress_signal["high"], anxiety_level["severe"]),
            ctrl.Rule(learned_signal["high"] & negative_sentiment["high"], anxiety_level["severe"]),
            ctrl.Rule(learned_signal["high"] & intensity_signal["medium"], anxiety_level["high"]),
            ctrl.Rule(learned_signal["high"] & protective_signal["low"], anxiety_level["high"]),
            ctrl.Rule(negative_sentiment["high"] & intensity_signal["high"], anxiety_level["high"]),
            ctrl.Rule(distress_signal["medium"] & negative_sentiment["medium"], anxiety_level["moderate"]),
            ctrl.Rule(distress_signal["medium"] & learned_signal["medium"], anxiety_level["high"]),
            ctrl.Rule(negative_sentiment["medium"] & learned_signal["medium"], anxiety_level["moderate"]),
            ctrl.Rule(distress_signal["low"] & learned_signal["low"] & negative_sentiment["low"], anxiety_level["low"]),
            ctrl.Rule(protective_signal["high"] & distress_signal["low"], anxiety_level["low"]),
            ctrl.Rule(protective_signal["high"] & learned_signal["low"], anxiety_level["low"]),
            ctrl.Rule(protective_signal["medium"] & negative_sentiment["medium"] & distress_signal["medium"], anxiety_level["moderate"]),
            ctrl.Rule(negative_sentiment["high"] & protective_signal["high"], anxiety_level["moderate"]),
            ctrl.Rule(intensity_signal["high"] & distress_signal["medium"], anxiety_level["high"]),
        ]

        self.anxiety_ctrl = ctrl.ControlSystem(rules)

    def _normalize_text(self, text):
        normalized = str(text)
        replacements = {
            "\u2018": "'",
            "\u2019": "'",
            "\u201c": '"',
            "\u201d": '"',
            "\n": " ",
            "\r": " ",
        }
        for source, target in replacements.items():
            normalized = normalized.replace(source, target)

        normalized = normalized.lower()
        normalized = re.sub(r"https?://\S+|www\.\S+", " ", normalized)
        normalized = re.sub(r"@\w+", " ", normalized)
        normalized = re.sub(r"#(\w+)", r" \1 ", normalized)
        normalized = normalized.replace("&amp;", " and ")
        normalized = re.sub(r"[^a-z0-9'!? ]+", " ", normalized)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized

    def _tokenize(self, normalized_text):
        return re.findall(r"[a-z]+(?:'[a-z]+)?", normalized_text)

    def _informative_terms(self, tokens):
        base_tokens = [
            token for token in tokens
            if len(token) >= 3 and token not in STOPWORDS
        ]
        bigrams = [
            f"{base_tokens[index]} {base_tokens[index + 1]}"
            for index in range(len(base_tokens) - 1)
        ]
        return set(base_tokens + bigrams)

    def _fit_dataset_signal(self, dataset_path):
        if not dataset_path.exists():
            return {}

        try:
            df = pd.read_csv(dataset_path, usecols=["tweet", "label"]).dropna(subset=["tweet", "label"])
        except Exception:
            return {}

        anxiety_counter = Counter()
        calm_counter = Counter()

        for row in df.itertuples(index=False):
            normalized = self._normalize_text(row.tweet)
            tokens = self._tokenize(normalized)
            doc_terms = self._informative_terms(tokens)
            if int(row.label) == 1:
                anxiety_counter.update(doc_terms)
            else:
                calm_counter.update(doc_terms)

        vocabulary = set(anxiety_counter) | set(calm_counter)
        if not vocabulary:
            return {}

        total_anxiety = sum(anxiety_counter.values())
        total_calm = sum(calm_counter.values())
        vocab_size = len(vocabulary)
        learned_weights = {}

        for term in vocabulary:
            frequency = anxiety_counter[term] + calm_counter[term]
            min_frequency = 2 if " " in term else 3
            if frequency < min_frequency:
                continue

            log_odds = math.log((anxiety_counter[term] + 1) / (total_anxiety + vocab_size))
            log_odds -= math.log((calm_counter[term] + 1) / (total_calm + vocab_size))

            if abs(log_odds) < 0.30:
                continue

            learned_weights[term] = log_odds

        sorted_terms = sorted(learned_weights.items(), key=lambda item: abs(item[1]), reverse=True)
        return dict(sorted_terms[:700])

    def _local_modifier(self, tokens, index):
        window = tokens[max(0, index - 3):index]
        modifier = 1.0
        if any(token in INTENSIFIERS for token in window[-2:]):
            modifier *= 1.25
        if any(token in NEGATORS for token in window):
            modifier *= -0.70
        return modifier

    def _weighted_token_score(self, tokens, weight_map):
        raw_score = 0.0
        for index, token in enumerate(tokens):
            if token in weight_map:
                raw_score += weight_map[token] * self._local_modifier(tokens, index)
        return raw_score

    def _weighted_phrase_score(self, normalized_text, phrase_weights):
        raw_score = 0.0
        for phrase, weight in phrase_weights.items():
            pattern = rf"(?<![a-z]){re.escape(phrase)}(?![a-z])"
            raw_score += len(re.findall(pattern, normalized_text)) * weight
        return raw_score

    def _scaled_positive_score(self, raw_value, scale):
        raw_value = max(0.0, raw_value)
        return float(min(10.0, 10.0 * (1.0 - math.exp(-raw_value / scale))))

    def _keyword_components(self, normalized_text, tokens):
        token_score = 0.0
        phrase_score = 0.0

        for keyword, weight in self.keyword_weights.items():
            normalized_keyword = self._normalize_text(keyword)
            if not normalized_keyword:
                continue
            if " " in normalized_keyword:
                phrase_score += self._weighted_phrase_score(normalized_text, {normalized_keyword: weight})
            else:
                term_weights = {normalized_keyword: weight}
                token_score += self._weighted_token_score(tokens, term_weights)

        raw_score = token_score + phrase_score
        return raw_score, self._scaled_positive_score(raw_score, scale=4.5)

    def _learned_signal_components(self, tokens):
        if not self.learned_signal_weights:
            return 0.0, 0.0, 0.0

        doc_terms = self._informative_terms(tokens)
        anxiety_raw = sum(
            self.learned_positive_terms[term]
            for term in doc_terms
            if term in self.learned_positive_terms
        )
        calm_raw = sum(
            self.learned_negative_terms[term]
            for term in doc_terms
            if term in self.learned_negative_terms
        )
        learned_score = self._scaled_positive_score(anxiety_raw, scale=3.8)
        calm_score = self._scaled_positive_score(calm_raw, scale=3.8)
        return anxiety_raw, learned_score, calm_score

    def _crisis_score(self, normalized_text):
        raw_score = self._weighted_phrase_score(normalized_text, CRISIS_PHRASES)
        if "suicide" in normalized_text or "suicidal" in normalized_text:
            raw_score += 4.0
        return self._scaled_positive_score(raw_score, scale=2.5)

    def _intensity_score(self, original_text, normalized_text, tokens):
        uppercase_words = re.findall(r"\b[A-Z]{3,}\b", str(original_text))
        elongated_words = re.findall(r"(.)\1{2,}", normalized_text)
        exclamations = normalized_text.count("!")
        questions = normalized_text.count("?")
        first_person_count = sum(token in FIRST_PERSON for token in tokens)
        intensifier_count = sum(token in INTENSIFIERS for token in tokens)

        raw_score = (
            1.0 * min(exclamations, 4)
            + 0.6 * min(questions, 3)
            + 1.2 * min(len(uppercase_words), 3)
            + 0.8 * min(len(elongated_words), 3)
            + 0.30 * min(first_person_count, 6)
            + 0.55 * min(intensifier_count, 4)
        )
        return float(min(10.0, raw_score))

    def _negative_sentiment_score(self, text):
        sentiment = self.analyzer.polarity_scores(text)
        negative_score = (sentiment["neg"] * 10.0) + (max(0.0, -sentiment["compound"]) * 6.0)
        return float(min(10.0, negative_score))

    def get_sentiment_score(self, text):
        return self.analyzer.polarity_scores(str(text))["compound"]

    def get_keyword_score(self, text):
        normalized_text = self._normalize_text(text)
        tokens = self._tokenize(normalized_text)
        _, keyword_score = self._keyword_components(normalized_text, tokens)
        return keyword_score

    def get_feature_breakdown(self, text):
        normalized_text = self._normalize_text(text)
        tokens = self._tokenize(normalized_text)

        keyword_raw, keyword_score = self._keyword_components(normalized_text, tokens)
        distress_raw = (
            self._weighted_token_score(tokens, DISTRESS_TERMS)
            + self._weighted_phrase_score(normalized_text, DISTRESS_PHRASES)
            + max(0.0, keyword_raw)
        )
        distress_signal = self._scaled_positive_score(distress_raw, scale=5.6)

        protective_raw = (
            self._weighted_token_score(tokens, PROTECTIVE_TERMS)
            + self._weighted_phrase_score(normalized_text, PROTECTIVE_PHRASES)
        )

        learned_raw, learned_signal, learned_calm_signal = self._learned_signal_components(tokens)
        protective_signal = self._scaled_positive_score(
            max(0.0, protective_raw) + learned_calm_signal * 0.35,
            scale=4.8,
        )

        negative_sentiment = self._negative_sentiment_score(text)
        intensity_signal = self._intensity_score(text, normalized_text, tokens)
        crisis_signal = self._crisis_score(normalized_text)

        heuristic_score = (
            0.28 * distress_signal
            + 0.22 * learned_signal
            + 0.17 * negative_sentiment
            + 0.14 * keyword_score
            + 0.09 * intensity_signal
            + 0.10 * crisis_signal
            - 0.18 * protective_signal
        )
        heuristic_score = float(max(0.0, min(10.0, heuristic_score)))

        return {
            "sentiment_score": self.get_sentiment_score(text),
            "negative_sentiment": negative_sentiment,
            "keyword_score": keyword_score,
            "distress_signal": distress_signal,
            "learned_signal": learned_signal,
            "protective_signal": protective_signal,
            "intensity_signal": intensity_signal,
            "crisis_signal": crisis_signal,
            "heuristic_score": heuristic_score,
        }

    def compute_prediction(self, text):
        if not isinstance(text, str):
            return 0.0

        features = self.get_feature_breakdown(text)

        anxiety_simulation = ctrl.ControlSystemSimulation(self.anxiety_ctrl)
        anxiety_simulation.input["negative_sentiment"] = features["negative_sentiment"]
        anxiety_simulation.input["distress_signal"] = features["distress_signal"]
        anxiety_simulation.input["learned_signal"] = features["learned_signal"]
        anxiety_simulation.input["intensity_signal"] = features["intensity_signal"]
        anxiety_simulation.input["protective_signal"] = features["protective_signal"]
        try:
            anxiety_simulation.compute()
            fuzzy_score = float(anxiety_simulation.output["anxiety_level"])
        except Exception:
            fuzzy_score = features["heuristic_score"]
        final_score = (0.68 * fuzzy_score) + (0.32 * features["heuristic_score"])

        if features["crisis_signal"] >= 6.0:
            final_score = max(final_score, 8.5)
        elif features["crisis_signal"] >= 3.0:
            final_score = max(final_score, 7.4)

        return float(max(0.0, min(10.0, final_score)))

    def interpret_anxiety_score(self, score):
        if score >= 8.5:
            return "Very High Anxiety"
        if score >= 6.7:
            return "High Anxiety"
        if score >= 4.3:
            return "Moderate Anxiety"
        return "Low Anxiety"
