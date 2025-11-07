import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import re
import logging

logger = logging.getLogger(__name__)

class BilingualSentimentAnalyzer:
    def __init__(self):
        # Jangan train model di __init__ - ini penyebab hang!
        self.vectorizers = {}
        self.models = {}
        self.is_trained = False
        self.training_data_setup = False
        
        # Setup training data (cepat, tanpa training)
        self._setup_training_data()
    
    def _setup_training_data(self):
        """Setup training data tanpa training models"""
        if self.training_data_setup:
            return
            
        # Minimal training data
        self.english_texts = [
            "I love this product!", "Excellent!", "Very satisfied!",
            "Terrible product!", "Very disappointed!", "Poor quality!",
            "It's okay.", "Average.", "Not bad not good."
        ]
        
        self.english_labels = ['positive', 'positive', 'positive', 
                              'negative', 'negative', 'negative',
                              'neutral', 'neutral', 'neutral']
        
        self.indonesian_texts = [
            "Saya suka produk ini!", "Bagus!", "Sangat puas!",
            "Produk jelek!", "Sangat kecewa!", "Kualitas buruk!",
            "Lumayanlah.", "Biasa saja.", "Cukup baik."
        ]
        
        self.indonesian_labels = ['positive', 'positive', 'positive',
                                 'negative', 'negative', 'negative', 
                                 'neutral', 'neutral', 'neutral']
        
        self.training_data_setup = True
    
    def _ensure_models_trained(self):
        """Train models hanya ketika pertama kali dibutuhkan"""
        if self.is_trained:
            return
            
        logger.info("Training models...")
        
        try:
            # English models
            self.vectorizers['english'] = TfidfVectorizer(max_features=500, stop_words='english')
            processed_english = [self._preprocess_text(text, 'english') for text in self.english_texts]
            X_en = self.vectorizers['english'].fit_transform(processed_english)
            
            self.models['english'] = {
                'NaiveBayes': MultinomialNB().fit(X_en, self.english_labels)
            }
            
            # Indonesian models  
            self.vectorizers['indonesian'] = TfidfVectorizer(max_features=500)
            processed_indonesian = [self._preprocess_text(text, 'indonesian') for text in self.indonesian_texts]
            X_id = self.vectorizers['indonesian'].fit_transform(processed_indonesian)
            
            self.models['indonesian'] = {
                'NaiveBayes': MultinomialNB().fit(X_id, self.indonesian_labels)
            }
            
            self.is_trained = True
            logger.info("Models trained successfully")
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            # Fallback: set trained anyway untuk avoid infinite loop
            self.is_trained = True
    
    def _preprocess_text(self, text, language):
        """Simple text preprocessing"""
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        return text
    
    def detect_language(self, text):
        """Fast language detection"""
        text_lower = text.lower()
        indo_words = ['yang', 'dan', 'di', 'ke', 'dari', 'ini', 'itu', 'saya', 'aku']
        english_words = ['the', 'and', 'to', 'of', 'a', 'in', 'is', 'it', 'you', 'i']
        
        id_count = sum(1 for word in indo_words if word in text_lower)
        en_count = sum(1 for word in english_words if word in text_lower)
        
        return 'indonesian' if id_count > en_count else 'english'
    
    def analyze_sentiment(self, text, method='NaiveBayes', language='auto'):
        """Analyze sentiment dengan lazy loading"""
        try:
            # Train models hanya ketika pertama kali dipanggil
            self._ensure_models_trained()
            
            # Detect language
            if language == 'auto':
                detected_lang = self.detect_language(text)
            else:
                detected_lang = language
            
            # Fallback ke NaiveBayes jika method tidak tersedia
            if method not in self.models.get(detected_lang, {}):
                method = 'NaiveBayes'
            
            # Preprocess dan predict
            processed_text = self._preprocess_text(text, detected_lang)
            vectorizer = self.vectorizers[detected_lang]
            model = self.models[detected_lang][method]
            
            X = vectorizer.transform([processed_text])
            prediction = model.predict(X)[0]
            probabilities = model.predict_proba(X)[0]
            confidence = max(probabilities)
            
            return {
                'sentiment_label': prediction,
                'confidence_score': float(confidence),
                'method_used': method,
                'language_detected': detected_lang,
                'processed_text': processed_text,
                'timestamp': pd.Timestamp.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Analysis error: {e}")
            return {
                'sentiment_label': 'neutral',
                'confidence_score': 0.5,
                'method_used': method,
                'language_detected': 'english',
                'error': str(e)
            }
