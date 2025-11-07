import streamlit as st
import time
import logging
import sys
import os
from datetime import datetime

# Fix import path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
if src_path not in sys.path:
    sys.path.append(src_path)

try:
    from database_manager import DatabaseManager
    from sentiment_analyzer import BilingualSentimentAnalyzer
    from config import db_config, app_config
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def init_session_state():
    """Initialize session state"""
    if 'backend' not in st.session_state:
        try:
            st.session_state.backend = SentilBackend()
        except Exception as e:
            st.error(f"Failed to initialize: {e}")
            return False
    
    if 'show_test' not in st.session_state:
        st.session_state.show_test = False
    if 'test_text' not in st.session_state:
        st.session_state.test_text = ""
    
    return True

class SentilBackend:
    def __init__(self):
        self.db = DatabaseManager()
        self.analyzer = BilingualSentimentAnalyzer()
        self.batch_size = app_config.processing_batch_size
        
        # Quick connection test
        try:
            if not self.db.test_connection():
                logger.warning("Database connection test failed")
        except:
            logger.warning("Database connection test skipped")

    def process_queue(self):
        """Process queued items dengan semua metode"""
        try:
            queued_items = self.db.get_queued_items(self.batch_size)
            
            if not queued_items:
                return 0
            
            processed_count = 0
            for item in queued_items:
                try:
                    slot_id = self.db.acquire_session_slot(item['tier'], item['user_id'])
                    if not slot_id:
                        continue
                    
                    self.db.update_queue_status(item['queue_id'], 'processing', slot_id)
                    
                    # Gunakan metode yang dipilih user
                    result = self.analyzer.analyze_sentiment(
                        item['input_text'], 
                        item.get('method', 'NaiveBayes'),
                        language='auto'
                    )
                    
                    self.db.insert_result(
                        queue_id=item['queue_id'],
                        sentiment_label=result['sentiment_label'],
                        confidence_score=result['confidence_score'],
                        json_result=result,
                        processed_by=f"Streamlit_{result['method_used']}_{result['language_detected']}"
                    )
                    
                    self.db.update_queue_status(item['queue_id'], 'done')
                    self.db.release_session_slot(slot_id)
                    processed_count += 1
                    
                    logger.info(f"Processed with {result['method_used']}: {result['sentiment_label']}")
                    
                except Exception as e:
                    logger.error(f"Failed to process {item['queue_id']}: {e}")
                    self.db.update_queue_status(item['queue_id'], 'error')
            
            return processed_count
            
        except Exception as e:
            logger.error(f"Queue processing error: {e}")
            return 0

def show_sidebar(backend):
    """Show sidebar dengan info methods"""
    st.sidebar.header("🔧 Configuration")
    
    # Connection info
    try:
        conn_info = db_config.parse_connection_string()
        if conn_info and 'error' not in conn_info:
            st.sidebar.success("✅ DB Connected")
    except:
        st.sidebar.info("🔧 Checking connection...")
    
    # Available methods info
    st.sidebar.subheader("📊 Available Methods")
    methods_info = {
        'NaiveBayes': 'Fast and simple',
        'KNN': 'Nearest neighbors',
        'RandomForest': 'Ensemble trees', 
        'SVM': 'Support Vector Machine'
    }
    
    for method, desc in methods_info.items():
        st.sidebar.write(f"**{method}**: {desc}")
    
    # Test examples
    st.sidebar.subheader("🧪 Test Examples")
    examples = {
        "English Positive": "I absolutely love this product! It's fantastic and works perfectly!",
        "English Negative": "This is terrible quality. Very disappointed with my purchase.",
        "Indonesian Positive": "Saya sangat suka produk ini! Kualitasnya bagus sekali dan harganya worth it!",
        "Indonesian Negative": "Kualitas sangat jelek. Sangat kecewa dengan pembelian ini."
    }
    
    for lang, text in examples.items():
        if st.sidebar.button(f"{lang}"):
            st.session_state.test_text = text
            st.session_state.show_test = True
            st.rerun()

def show_test_section(backend):
    """Show test section dengan semua metode"""
    if not st.session_state.show_test:
        return
    
    st.subheader("🧪 Multi-Method Analysis")
    
    # Input text
    test_text = st.text_area(
        "Text to analyze:", 
        st.session_state.test_text or "Type your text here...",
        height=100
    )
    
    # Method selection dengan deskripsi
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("⚙️ Analysis Settings")
        
        method = st.selectbox(
            "Machine Learning Method:",
            options=['NaiveBayes', 'KNN', 'RandomForest', 'SVM'],
            index=0,
            help="Pilih metode machine learning untuk analisis"
        )
        
        # Method descriptions
        method_descriptions = {
            'NaiveBayes': '🔹 Cepat dan efisien untuk text classification',
            'KNN': '🔹 Berdasarkan similarity dengan training data', 
            'RandomForest': '🔹 Ensemble method dengan multiple decision trees',
            'SVM': '🔹 Powerful untuk high-dimensional data'
        }
        
        st.info(method_descriptions[method])
    
    with col2:
        st.subheader("🌐 Language Settings")
        
        language = st.selectbox(
            "Language Processing:",
            options=['auto', 'english', 'indonesian'],
            index=0,
            help="Auto: deteksi otomatis, English/Indonesian: paksa bahasa tertentu"
        )
        
        language_info = {
            'auto': '🔸 Deteksi bahasa secara otomatis',
            'english': '🔸 Proses sebagai teks English',
            'indonesian': '🔸 Proses sebagai teks Indonesia'
        }
        
        st.info(language_info[language])
    
    # Analysis button
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if st.button("🚀 Analyze Sentiment", type="primary", use_container_width=True):
            if not test_text.strip():
                st.warning("⚠️ Please enter some text to analyze")
                return
                
            with st.spinner(f"Analyzing with {method}..."):
                try:
                    start_time = time.time()
                    result = backend.analyzer.analyze_sentiment(test_text, method, language)
                    analysis_time = time.time() - start_time
                    
                    # Display results
                    show_analysis_results(result, analysis_time, test_text)
                    
                except Exception as e:
                    st.error(f"❌ Analysis failed: {e}")
    
    with col2:
        if st.button("❌ Close", use_container_width=True):
            st.session_state.show_test = False
            st.rerun()

def show_analysis_results(result, analysis_time, original_text):
    """Display analysis results secara comprehensive"""
    st.success("✅ Analysis Complete!")
    
    # Main metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        sentiment = result['sentiment_label']
        emoji = "😊" if sentiment == 'positive' else "😞" if sentiment == 'negative' else "😐"
        st.metric("Sentiment", f"{emoji} {sentiment}")
    
    with col2:
        confidence = result['confidence_score']
        color = "green" if confidence > 0.7 else "orange" if confidence > 0.5 else "red"
        st.metric("Confidence", f"{confidence:.1%}")
    
    with col3:
        method = result['method_used']
        st.metric("Method", f"📊 {method}")
    
    with col4:
        lang = result['language_detected']
        flag = "🇺🇸" if lang == 'english' else "🇮🇩"
        st.metric("Language", f"{flag} {lang}")
    
    # Performance info
    st.info(f"⏱️ Analysis took {analysis_time:.2f} seconds | 📝 {result.get('word_count', 0)} words")
    
    # Detailed results
    with st.expander("📋 Detailed Analysis Results"):
        tab1, tab2, tab3 = st.tabs(["Result JSON", "Text Info", "Method Info"])
        
        with tab1:
            st.json(result)
        
        with tab2:
            st.write(f"**Original Text:** {original_text}")
            st.write(f"**Processed Text:** {result.get('processed_text', 'N/A')}")
            st.write(f"**Text Length:** {result.get('text_length', 0)} characters")
            st.write(f"**Word Count:** {result.get('word_count', 0)} words")
        
        with tab3:
            st.write(f"**Available Methods:** {', '.join(result.get('available_methods', []))}")
            st.write(f"**Used Method:** {result['method_used']}")
            st.write(f"**Language Detected:** {result['language_detected']}")

def main():
    """Main app function"""
    st.set_page_config(
        page_title="Sentil - Multi-Method Analyzer",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 Sentil Multi-Method Sentiment Analyzer")
    st.markdown("**Naive Bayes • KNN • Random Forest • SVM** - Bilingual Analysis")
    
    # Initialize
    if not init_session_state():
        st.stop()
    
    backend = st.session_state.backend
    
    # Sidebar
    show_sidebar(backend)
    
    # Quick actions
    st.subheader("🚀 Quick Actions")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🔄 Process Queue", type="primary", use_container_width=True):
            with st.spinner("Processing queue..."):
                count = backend.process_queue()
                if count > 0:
                    st.success(f"✅ Processed {count} items")
                else:
                    st.info("📭 No items in queue")
    
    with col2:
        if st.button("📊 Queue Status", use_container_width=True):
            try:
                items = backend.db.get_queued_items(5)
                st.info(f"📋 Queued items: {len(items)}")
                if items:
                    for item in items:
                        st.write(f"- `{item['method']}`: {item['input_text'][:30]}...")
            except Exception as e:
                st.error(f"❌ Failed: {e}")
    
    with col3:
        if st.button("🧪 Test Analysis", type="secondary", use_container_width=True):
            st.session_state.show_test = True
            st.rerun()
    
    with col4:
        if st.button("🆕 New Analysis", use_container_width=True):
            st.session_state.test_text = ""
            st.session_state.show_test = True
            st.rerun()
    
    # Test section
    show_test_section(backend)
    
    # Methods overview
    st.markdown("---")
    st.subheader("🔬 Supported Methods Overview")
    
    methods_cols = st.columns(4)
    methods_info = [
        ("Naive Bayes", "Statistical method based on Bayes theorem", "🎯"),
        ("K-Nearest Neighbors", "Instance-based learning", "📈"),
        ("Random Forest", "Ensemble of decision trees", "🌳"),
        ("Support Vector Machine", "Maximizes margin between classes", "⚡")
    ]
    
    for idx, (name, desc, icon) in enumerate(methods_info):
        with methods_cols[idx]:
            st.metric(f"{icon} {name}", desc)
    
    # Footer
    st.markdown("---")
    st.caption("Sentil v1.0 | Multi-Method Bilingual Sentiment Analysis | Streamlit + Neon DB")

if __name__ == "__main__":
    main()
