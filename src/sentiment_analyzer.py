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

# Download NLTK data (cached)
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    pass

logger = logging.getLogger(__name__)

class BilingualSentimentAnalyzer:
    def __init__(self):
        self.vectorizers = {
            'english': TfidfVectorizer(max_features=1000, stop_words='english'),  # Reduced features
            'indonesian': TfidfVectorizer(max_features=1000)
        }
        
        # Gunakan hanya NaiveBayes untuk faster training
        self.models = {
            'english': {
                'NaiveBayes': MultinomialNB(),
                # 'KNN': KNeighborsClassifier(n_neighbors=3),  # Simplified untuk performance
                # 'RandomForest': RandomForestClassifier(n_estimators=50),  # Reduced trees
            },
            'indonesian': {
                'NaiveBayes': MultinomialNB(),
                # 'KNN': KNeighborsClassifier(n_neighbors=3),
                # 'RandomForest': RandomForestClassifier(n_estimators=50),
            }
        }
        
        # Indonesian stopwords manual (simplified)
        self.indonesian_stopwords = {
            'yang', 'dan', 'di', 'ke', 'dari', 'untuk', 'dengan', 'ini', 'itu', 
            'tidak', 'aku', 'kamu', 'kami', 'mereka', 'adalah', 'ada', 'sudah', 
            'akan', 'telah', 'pada', 'juga', 'dalam', 'bahwa', 'atau', 'juga'
        }
        
        self.is_trained = False
        self._setup_minimal_training_data()  # Fast training data
    
    def _setup_minimal_training_data(self):
        """Setup minimal training data for quick startup"""
        
        # Minimal English training data (6 samples each)
        self.english_texts = [
            # Positive
            "I love this product!", "Excellent quality!", "Very satisfied!",
            # Negative
            "Terrible product!", "Very disappointed!", "Poor quality!",
            # Neutral  
            "It's okay.", "Average product.", "Not bad not good."
        ]
        
        self.english_labels = ['positive', 'positive', 'positive', 
                              'negative', 'negative', 'negative',
                              'neutral', 'neutral', 'neutral']
        
        # Minimal Indonesian training data
        self.indonesian_texts = [
            # Positive
            "Saya suka produk ini!", "Kualitas bagus!", "Sangat puas!",
            # Negative
            "Produk jelek!", "Sangat kecewa!", "Kualitas buruk!",
            # Neutral
            "Lumayanlah.", "Biasa saja.", "Tidak jelek tidak bagus."
        ]
        
        self.indonesian_labels = ['positive', 'positive', 'positive',
                                 'negative', 'negative', 'negative', 
                                 'neutral', 'neutral', 'neutral']
        
        # Train models immediately
        self._train_fast_models()
    
    def _train_fast_models(self):
        """Fast training dengan minimal data"""
        try:
            # Train English models
            processed_english = [self._preprocess_english_fast(text) for text in self.english_texts]
            X_en = self.vectorizers['english'].fit_transform(processed_english)
            y_en = self.english_labels
            
            for name, model in self.models['english'].items():
                model.fit(X_en, y_en)
                logger.info(f"✅ English model {name} trained")
            
            # Train Indonesian models
            processed_indonesian = [self._preprocess_indonesian_fast(text) for text in self.indonesian_texts]
            X_id = self.vectorizers['indonesian'].fit_transform(processed_indonesian)
            y_id = self.indonesian_labels
            
            for name, model in self.models['indonesian'].items():
                model.fit(X_id, y_id)
                logger.info(f"✅ Indonesian model {name} trained")
            
            self.is_trained = True
            logger.info("🎯 All models trained successfully")
            
        except Exception as e:
            logger.error(f"❌ Model training failed: {e}")
            # Fallback: tetap set is_trained untuk avoid infinite loop
            self.is_trained = True
    
    def _preprocess_english_fast(self, text):
        """Fast English preprocessing"""
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        return text
    
    def _preprocess_indonesian_fast(self, text):
        """Fast Indonesian preprocessing"""
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        return text
    
    def detect_language(self, text: str) -> str:
        """Fast language detection"""
        text_lower = text.lower()
        
        # Simple keyword detection
        indo_words = ['yang', 'dan', 'di', 'ke', 'dari', 'untuk', 'dengan', 'ini', 'itu']
        english_words = ['the', 'and', 'to', 'of', 'a', 'in', 'is', 'it', 'you']
        
        id_count = sum(1 for word in indo_words if word in text_lower)
        en_count = sum(1 for word in english_words if word in text_lower)
        
        return 'indonesian' if id_count > en_count else 'english'
    
    def analyze_sentiment(self, text: str, method: str = 'NaiveBayes', language: str = 'auto') -> dict:
        """Fast sentiment analysis"""
        try:
            # Auto-detect language if not specified
            if language == 'auto':
                detected_lang = self.detect_language(text)
            else:
                detected_lang = language
            
            # Ensure we have a valid method
            if method not in self.models[detected_lang]:
                method = 'NaiveBayes'  # Fallback to NaiveBayes
            
            # Preprocess based on language
            if detected_lang == 'english':
                processed_text = self._preprocess_english_fast(text)
                vectorizer = self.vectorizers['english']
                model = self.models['english'][method]
            else:
                processed_text = self._preprocess_indonesian_fast(text)
                vectorizer = self.vectorizers['indonesian']
                model = self.models['indonesian'][method]
            
            # Vectorize and predict
            X = vectorizer.transform([processed_text])
            prediction = model.predict(X)[0]
            probabilities = model.predict_proba(X)[0]
            confidence = max(probabilities)
            
            # Simple polarity estimation
            positive_words = ['love', 'good', 'great', 'excellent', 'suka', 'bagus', 'puas', 'senang']
            negative_words = ['terrible', 'bad', 'poor', 'disappointed', 'jelek', 'buruk', 'kecewa']
            
            text_lower = text.lower()
            positive_count = sum(1 for word in positive_words if word in text_lower)
            negative_count = sum(1 for word in negative_words if word in text_lower)
            
            if positive_count > negative_count:
                polarity = 0.5
            elif negative_count > positive_count:
                polarity = -0.5
            else:
                polarity = 0.0
            
            result = {
                'sentiment_label': prediction,
                'confidence_score': float(confidence),
                'method_used': method,
                'language_detected': detected_lang,
                'polarity': round(polarity, 3),
                'processed_text': processed_text,
                'timestamp': pd.Timestamp.now().isoformat()
            }
            
            logger.info(f"✅ Analysis complete: {prediction}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Analysis failed: {e}")
            # Return fallback result
            return {
                'sentiment_label': 'neutral',
                'confidence_score': 0.5,
                'method_used': method,
                'language_detected': 'english',
                'error': str(e)
            }
