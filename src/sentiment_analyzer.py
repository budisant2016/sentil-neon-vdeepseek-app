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
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# Download NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    pass

logger = logging.getLogger(__name__)

class BilingualSentimentAnalyzer:
    def __init__(self):
        self.vectorizers = {
            'english': TfidfVectorizer(max_features=5000, stop_words='english'),
            'indonesian': TfidfVectorizer(max_features=5000)
        }
        self.models = {
            'english': {
                'NaiveBayes': MultinomialNB(),
                'KNN': KNeighborsClassifier(n_neighbors=5),
                'RandomForest': RandomForestClassifier(n_estimators=100),
                'SVM': SVC(probability=True)
            },
            'indonesian': {
                'NaiveBayes': MultinomialNB(),
                'KNN': KNeighborsClassifier(n_neighbors=5),
                'RandomForest': RandomForestClassifier(n_estimators=100),
                'SVM': SVC(probability=True)
            }
        }
        
        # Setup Sastrawi untuk Bahasa Indonesia
        self.indonesian_stemmer = StemmerFactory().create_stemmer()
        self.indonesian_stopwords = StopWordRemoverFactory().get_stop_words()
        
        self.is_trained = False
        self.setup_bilingual_training_data()
    
    def detect_language(self, text: str) -> str:
        """Detect language of the text"""
        text_lower = text.lower()
        
        # Simple keyword-based detection
        indonesian_keywords = ['yang', 'dan', 'di', 'ke', 'dari', 'untuk', 'dengan', 'ini', 'itu', 'tidak', 'aku', 'kamu', 'kami', 'mereka', 'adalah', 'ada', 'sudah', 'akan', 'telah']
        english_keywords = ['the', 'and', 'to', 'of', 'a', 'in', 'is', 'it', 'you', 'that', 'he', 'was', 'for', 'on', 'are', 'as', 'with', 'his', 'they', 'i']
        
        id_count = sum(1 for word in indonesian_keywords if word in text_lower)
        en_count = sum(1 for word in english_keywords if word in text_lower)
        
        # Check for specific Indonesian characters/patterns
        if 'yg ' in text_lower or ' dgn ' in text_lower or ' tdk ' in text_lower:
            return 'indonesian'
        
        if id_count > en_count:
            return 'indonesian'
        elif en_count > id_count:
            return 'english'
        else:
            # Default to English if uncertain
            return 'english'
    
    def preprocess_english(self, text):
        """Preprocess English text"""
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
        return ' '.join(tokens)
    
    def preprocess_indonesian(self, text):
        """Preprocess Indonesian text"""
        text = str(text).lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove Indonesian stopwords
        tokens = [token for token in tokens if token not in self.indonesian_stopwords]
        
        # Stemming
        stemmed_tokens = [self.indonesian_stemmer.stem(token) for token in tokens]
        
        return ' '.join(stemmed_tokens)
    
    def setup_bilingual_training_data(self):
        """Setup training data for both English and Indonesian"""
        
        # English training data
        self.english_texts = [
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
        
        self.english_labels = ['positive'] * 10 + ['negative'] * 10 + ['neutral'] * 10
        
        # Indonesian training data
        self.indonesian_texts = [
            # Positive sentiments
            "Saya sangat suka produk ini, luar biasa!", "Pengalaman yang fantastis!",
            "Produk yang bagus, sangat direkomendasikan!", "Harga sesuai dengan kualitas.",
            "Puas dengan pembelian ini.", "Kualitas dan pelayanan yang outstanding!",
            "Pengalaman yang menyenangkan, akan beli lagi!", "Sempurna, persis yang saya butuhkan!",
            "Produk bagus dengan pengiriman cepat!", "Sangat senang dengan pembelian ini!",
            
            # Negative sentiments
            "Ini buruk sekali, pembelian terjelek.", "Sangat kecewa dengan kualitasnya.",
            "Pelayanan pelanggan yang buruk.", "Bukan yang saya harapkan, cukup jelek.",
            "Buang-buang uang, jangan beli ini.", "Produk mengerikan, rusak saat datang.",
            "Sangat tidak puas dengan ini.", "Kualitas yang jelek, sama sekali tidak berguna.",
            "Menyesal membeli produk ini.", "Pengalaman yang menyedihkan.",
            
            # Neutral sentiments
            "Lumayanlah, tidak istimewa.", "Produknya biasa saja.",
            "Tidak jelek, tapi tidak bagus juga.", "Kualitas biasa untuk harganya.",
            "Berfungsi seperti yang diharapkan, tidak lebih.", "Produk rata-rata, cukup bekerja.",
            "Tidak baik juga tidak buruk.", "Bisa diterima tapi bisa lebih baik.",
            "Kualitas standar, memenuhi kebutuhan dasar.", "Produk yang wajar untuk harganya."
        ]
        
        self.indonesian_labels = ['positive'] * 10 + ['negative'] * 10 + ['neutral'] * 10
        
        self.train_bilingual_models()
    
    def train_bilingual_models(self):
        """Train models for both languages"""
        try:
            # Train English models
            processed_english = [self.preprocess_english(text) for text in self.english_texts]
            X_en = self.vectorizers['english'].fit_transform(processed_english)
            y_en = self.english_labels
            
            for name, model in self.models['english'].items():
                model.fit(X_en, y_en)
                logger.info(f"✅ English model {name} trained successfully")
            
            # Train Indonesian models
            processed_indonesian = [self.preprocess_indonesian(text) for text in self.indonesian_texts]
            X_id = self.vectorizers['indonesian'].fit_transform(processed_indonesian)
            y_id = self.indonesian_labels
            
            for name, model in self.models['indonesian'].items():
                model.fit(X_id, y_id)
                logger.info(f"✅ Indonesian model {name} trained successfully")
            
            self.is_trained = True
            logger.info("🎯 All bilingual models trained successfully")
            
        except Exception as e:
            logger.error(f"❌ Bilingual model training failed: {e}")
            raise
    
    def analyze_sentiment(self, text: str, method: str = 'NaiveBayes', language: str = 'auto') -> dict:
        """Analyze sentiment with automatic language detection"""
        if not self.is_trained:
            self.train_bilingual_models()
        
        try:
            # Auto-detect language if not specified
            if language == 'auto':
                detected_lang = self.detect_language(text)
            else:
                detected_lang = language
            
            # Preprocess based on language
            if detected_lang == 'english':
                processed_text = self.preprocess_english(text)
                vectorizer = self.vectorizers['english']
                model = self.models['english'].get(method, self.models['english']['NaiveBayes'])
            else:  # indonesian
                processed_text = self.preprocess_indonesian(text)
                vectorizer = self.vectorizers['indonesian']
                model = self.models['indonesian'].get(method, self.models['indonesian']['NaiveBayes'])
            
            # Vectorize and predict
            X = vectorizer.transform([processed_text])
            prediction = model.predict(X)[0]
            probabilities = model.predict_proba(X)[0]
            confidence = max(probabilities)
            
            # Additional analysis with TextBlob (for English)
            if detected_lang == 'english':
                blob = TextBlob(text)
                polarity = blob.sentiment.polarity
                subjectivity = blob.sentiment.subjectivity
            else:
                # Simple polarity estimation for Indonesian
                positive_words = ['bagus', 'baik', 'suka', 'puas', 'senang', 'mantap', 'keren', 'luar biasa', 'fantastis']
                negative_words = ['jelek', 'buruk', 'kecewa', 'menyesal', 'susah', 'sulit', 'tidak suka', 'gagal']
                
                positive_count = sum(1 for word in positive_words if word in text.lower())
                negative_count = sum(1 for word in negative_words if word in text.lower())
                
                if positive_count > negative_count:
                    polarity = 0.5
                elif negative_count > positive_count:
                    polarity = -0.5
                else:
                    polarity = 0.0
                subjectivity = 0.5  # Default subjectivity for Indonesian
            
            result = {
                'sentiment_label': prediction,
                'confidence_score': float(confidence),
                'method_used': method,
                'language_detected': detected_lang,
                'textblob_polarity': round(polarity, 3),
                'textblob_subjectivity': round(subjectivity, 3) if detected_lang == 'english' else 'N/A',
                'processed_text': processed_text,
                'original_text': text[:100] + '...' if len(text) > 100 else text,
                'timestamp': pd.Timestamp.now().isoformat()
            }
            
            logger.info(f"✅ {detected_lang.upper()} analysis complete: {prediction} (confidence: {confidence:.2f})")
            return result
            
        except Exception as e:
            logger.error(f"❌ Bilingual sentiment analysis failed: {e}")
            return {
                'sentiment_label': 'error',
                'confidence_score': 0.0,
                'method_used': method,
                'language_detected': 'unknown',
                'error': str(e)
            }
    
    def batch_analyze(self, texts: list, method: str = 'NaiveBayes', language: str = 'auto') -> list:
        """Analyze multiple texts"""
        results = []
        for text in texts:
            result = self.analyze_sentiment(text, method, language)
            results.append(result)
        return results
