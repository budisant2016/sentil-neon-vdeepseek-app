import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from textblob import TextBlob
import logging

# Download NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    pass

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.models = {
            'NaiveBayes': MultinomialNB(),
            'KNN': KNeighborsClassifier(n_neighbors=5),
            'RandomForest': RandomForestClassifier(n_estimators=100),
            'SVM': SVC(probability=True)
        }
        self.is_trained = False
        self.setup_sample_data()
    
    def setup_sample_data(self):
        """Setup sample training data for demo purposes"""
        # Extended sample data for better accuracy
        self.sample_texts = [
            # Positive sentiments
            "I love this product, it's amazing!", "Absolutely fantastic experience!",
            "Excellent product, highly recommended!", "Good value for money.",
            "Satisfied with my purchase.", "Outstanding quality and service!",
            "Wonderful experience, will buy again!", "Perfect, exactly what I needed!",
            "Great product with fast delivery!", "Very happy with this purchase!",
            
            # Negative sentiments  
            "This is terrible, worst purchase ever.", "Very disappointed with the quality.",
            "Poor customer service.", "Not what I expected, quite bad.",
            "Waste of money, don't buy this.", "Horrible product, broken on arrival.",
            "Extremely dissatisfied with this.", "Awful quality, completely useless.",
            "Regret buying this product.", "Terrible experience overall.",
            
            # Neutral sentiments
            "It's okay, nothing special.", "The product is average.",
            "Not bad, but not great either.", "Mediocre quality for the price.",
            "It works as expected, nothing more.", "Average product, does the job.",
            "Neither good nor bad.", "Acceptable but could be better.",
            "Standard quality, meets basic needs.", "Fair product for the price."
        ]
        
        self.sample_labels = (
            ['positive'] * 10 + ['negative'] * 10 + ['neutral'] * 10
        )
        
        self.train_models()
    
    def preprocess_text(self, text):
        """Preprocess text for analysis"""
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
        return ' '.join(tokens)
    
    def train_models(self):
        """Train all ML models"""
        try:
            processed_texts = [self.preprocess_text(text) for text in self.sample_texts]
            X = self.vectorizer.fit_transform(processed_texts)
            y = self.sample_labels
            
            for name, model in self.models.items():
                model.fit(X, y)
                logger.info(f"✅ Model {name} trained successfully")
            
            self.is_trained = True
            
        except Exception as e:
            logger.error(f"❌ Model training failed: {e}")
            raise
    
    def analyze_sentiment(self, text: str, method: str = 'NaiveBayes') -> dict:
        """Analyze sentiment using specified method"""
        if not self.is_trained:
            self.train_models()
        
        try:
            processed_text = self.preprocess_text(text)
            X = self.vectorizer.transform([processed_text])
            model = self.models.get(method, self.models['NaiveBayes'])
            
            prediction = model.predict(X)[0]
            probabilities = model.predict_proba(X)[0]
            confidence = max(probabilities)
            
            # TextBlob analysis for additional insights
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            
            result = {
                'sentiment_label': prediction,
                'confidence_score': float(confidence),
                'method_used': method,
                'textblob_polarity': round(polarity, 3),
                'textblob_subjectivity': round(subjectivity, 3),
                'processed_text': processed_text,
                'timestamp': pd.Timestamp.now().isoformat()
            }
            
            logger.info(f"✅ Analysis complete: {prediction} (confidence: {confidence:.2f})")
            return result
            
        except Exception as e:
            logger.error(f"❌ Sentiment analysis failed: {e}")
            return {
                'sentiment_label': 'error',
                'confidence_score': 0.0,
                'method_used': method,
                'error': str(e)
            }
