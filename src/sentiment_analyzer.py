import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import re
import logging

logger = logging.getLogger(__name__)

class BilingualSentimentAnalyzer:
    def __init__(self):
        # Jangan train model di __init__
        self.vectorizers = {}
        self.models = {}
        self.is_trained = False
        self.training_data_setup = False
        
        # Setup training data (cepat)
        self._setup_training_data()
    
    def _setup_training_data(self):
        """Setup training data untuk semua metode"""
        if self.training_data_setup:
            return
            
        # Extended training data untuk akurasi yang lebih baik
        self.english_texts = [
            # Positive (12 samples)
            "I love this product! It's amazing!", "Excellent quality and fast delivery!", 
            "Very satisfied with my purchase!", "Outstanding service and support!",
            "Great value for money!", "Perfect, exactly what I needed!",
            "Wonderful experience!", "Highly recommended!", 
            "Fantastic product!", "Awesome features!", "Super happy!", "Best purchase ever!",
            
            # Negative (12 samples)  
            "Terrible product! Waste of money.", "Very disappointed with the quality.",
            "Poor customer service.", "Not what I expected, quite bad.",
            "Horrible experience!", "Broken on arrival.", 
            "Extremely dissatisfied!", "Awful quality!", "Regret buying this.",
            "Worst purchase!", "Does not work properly!", "Complete garbage!",
            
            # Neutral (6 samples)
            "It's okay, nothing special.", "The product is average.",
            "Not bad, but not great either.", "Mediocre quality.",
            "It works as expected.", "Standard product, meets basic needs."
        ]
        
        self.english_labels = (
            ['positive'] * 12 + 
            ['negative'] * 12 + 
            ['neutral'] * 6
        )
        
        self.indonesian_texts = [
            # Positive (12 samples)
            "Saya sangat suka produk ini! Luar biasa!", "Kualitas bagus dan pengiriman cepat!", 
            "Sangat puas dengan pembelian!", "Pelayanan dan dukungan yang outstanding!",
            "Harga sesuai kualitas!", "Sempurna, persis yang saya butuhkan!",
            "Pengalaman yang menyenangkan!", "Sangat direkomendasikan!", 
            "Produk yang fantastis!", "Fitur yang mengagumkan!", "Sangat senang!", "Pembelian terbaik!",
            
            # Negative (12 samples)
            "Produk yang jelek! Buang-buang uang.", "Sangat kecewa dengan kualitasnya.",
            "Pelayanan pelanggan yang buruk.", "Bukan yang saya harapkan, cukup jelek.",
            "Pengalaman yang mengerikan!", "Rusak saat datang.", 
            "Sangat tidak puas!", "Kualitas yang buruk!", "Menyesal membeli ini.",
            "Pembelian terburuk!", "Tidak bekerja dengan baik!", "Sampah!",
            
            # Neutral (6 samples)
            "Lumayanlah, tidak istimewa.", "Produknya biasa saja.",
            "Tidak jelek, tapi tidak bagus juga.", "Kualitas mediocre.",
            "Berfungsi seperti yang diharapkan.", "Produk standar, memenuhi kebutuhan dasar."
        ]
        
        self.indonesian_labels = (
            ['positive'] * 12 + 
            ['negative'] * 12 + 
            ['neutral'] * 6
        )
        
        self.training_data_setup = True
    
    def _ensure_models_trained(self, language):
        """Train models untuk bahasa tertentu hanya ketika dibutuhkan"""
        if language in self.models and self.models[language]:
            return
            
        logger.info(f"Training {language} models...")
        
        try:
            if language == 'english':
                texts = self.english_texts
                labels = self.english_labels
                stop_words = 'english'
            else:
                texts = self.indonesian_texts
                labels = self.indonesian_labels
                stop_words = None
            
            # Vectorizer
            self.vectorizers[language] = TfidfVectorizer(
                max_features=800,  # Balance antara performance dan accuracy
                stop_words=stop_words,
                ngram_range=(1, 2)  # Unigram dan bigram
            )
            
            processed_texts = [self._preprocess_text(text, language) for text in texts]
            X = self.vectorizers[language].fit_transform(processed_texts)
            
            # Train semua models dengan parameter optimized
            self.models[language] = {
                'NaiveBayes': MultinomialNB().fit(X, labels),
                'KNN': KNeighborsClassifier(
                    n_neighbors=5,  # Reduced untuk performance
                    weights='distance'
                ).fit(X, labels),
                'RandomForest': RandomForestClassifier(
                    n_estimators=50,  # Reduced dari 100
                    max_depth=10,     # Limit depth
                    random_state=42,
                    n_jobs=-1         # Use all cores
                ).fit(X, labels),
                'SVM': SVC(
                    kernel='linear',  # Linear kernel lebih cepat
                    probability=True,
                    random_state=42,
                    C=1.0             # Regularization parameter
                ).fit(X, labels)
            }
            
            logger.info(f"{language} models trained successfully")
            
        except Exception as e:
            logger.error(f"{language} model training failed: {e}")
            # Fallback ke NaiveBayes saja
            self._create_fallback_model(language, texts, labels)
    
    def _create_fallback_model(self, language, texts, labels):
        """Create fallback model jika training gagal"""
        try:
            processed_texts = [self._preprocess_text(text, language) for text in texts]
            X = TfidfVectorizer(max_features=500).fit_transform(processed_texts)
            
            self.models[language] = {
                'NaiveBayes': MultinomialNB().fit(X, labels)
            }
            logger.info(f"Fallback {language} model created")
        except:
            # Ultimate fallback - dummy model
            from sklearn.dummy import DummyClassifier
            self.models[language] = {
                'NaiveBayes': DummyClassifier(strategy='stratified').fit([[0]], ['neutral'])
            }
    
    def _preprocess_text(self, text, language):
        """Fast text preprocessing"""
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        return text
    
    def detect_language(self, text):
        """Fast language detection"""
        text_lower = text.lower()
        
        indo_words = ['yang', 'dan', 'di', 'ke', 'dari', 'ini', 'itu', 'saya', 'aku', 'kamu', 'kami', 'dengan', 'untuk']
        english_words = ['the', 'and', 'to', 'of', 'a', 'in', 'is', 'it', 'you', 'i', 'this', 'that', 'with', 'for']
        
        id_count = sum(1 for word in indo_words if word in text_lower)
        en_count = sum(1 for word in english_words if word in text_lower)
        
        return 'indonesian' if id_count > en_count else 'english'
    
    def get_available_methods(self, language='english'):
        """Get list of available methods untuk UI"""
        if language not in self.models:
            self._ensure_models_trained(language)
        
        return list(self.models.get(language, {}).keys())
    
    def analyze_sentiment(self, text, method='NaiveBayes', language='auto'):
        """Analyze sentiment dengan semua metode yang tersedia"""
        try:
            # Detect language
            if language == 'auto':
                detected_lang = self.detect_language(text)
            else:
                detected_lang = language
            
            # Train models untuk bahasa ini jika belum
            self._ensure_models_trained(detected_lang)
            
            # Validasi method
            available_methods = self.get_available_methods(detected_lang)
            if method not in available_methods:
                method = available_methods[0]  # Fallback ke method pertama
            
            # Preprocess dan predict
            processed_text = self._preprocess_text(text, detected_lang)
            vectorizer = self.vectorizers[detected_lang]
            model = self.models[detected_lang][method]
            
            X = vectorizer.transform([processed_text])
            prediction = model.predict(X)[0]
            
            # Confidence score
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(X)[0]
                confidence = max(probabilities)
            else:
                confidence = 0.7  # Default confidence
            
            # Additional metrics
            text_length = len(text)
            word_count = len(text.split())
            
            return {
                'sentiment_label': prediction,
                'confidence_score': float(confidence),
                'method_used': method,
                'language_detected': detected_lang,
                'processed_text': processed_text,
                'text_length': text_length,
                'word_count': word_count,
                'available_methods': available_methods,
                'timestamp': pd.Timestamp.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Analysis error: {e}")
            return {
                'sentiment_label': 'neutral',
                'confidence_score': 0.5,
                'method_used': method,
                'language_detected': 'english',
                'error': str(e),
                'available_methods': ['NaiveBayes']  # Fallback
            }
