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

def show_batch_analysis_section(backend):
    """Show batch analysis input form"""
    st.subheader("📚 Batch Text Analysis")
    
    with st.form("batch_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Batch Input**")
            batch_texts = st.text_area(
                "Enter texts (one per line):",
                placeholder="I love this product!\nThis is terrible!\nIt's okay...",
                height=200,
                help="Enter one text per line. Maximum texts based on your tier."
            )
            
            # Parse texts
            texts = [text.strip() for text in batch_texts.split('\n') if text.strip()]
            text_count = len(texts)
            
            st.write(f"📊 Texts detected: {text_count}")
        
        with col2:
            st.write("**Analysis Settings**")
            user_tier = st.selectbox(
                "Your Tier:",
                options=[1, 2, 3],
                index=0,
                format_func=lambda x: f"Tier {x} - {'Guest (10 max)' if x == 1 else 'Registered (30 max)' if x == 2 else 'Premium (100 max)'}"
            )
            
            method = st.selectbox(
                "Analysis Method:",
                options=['NaiveBayes', 'KNN', 'RandomForest', 'SVM']
            )
            
            language = st.selectbox(
                "Language:",
                options=['auto', 'english', 'indonesian']
            )
            
            # Show tier limits
            tier_limits = {1: 10, 2: 30, 3: 100}
            max_limit = tier_limits[user_tier]
            
            if text_count > 0:
                if text_count > max_limit:
                    st.error(f"❌ Tier {user_tier} limit exceeded: {text_count}/{max_limit} texts")
                else:
                    st.success(f"✅ Within tier limit: {text_count}/{max_limit} texts")
        
        submitted = st.form_submit_button("🚀 Submit Batch Analysis", type="primary")
        
        if submitted:
            if not texts:
                st.error("⚠️ Please enter some texts to analyze")
                return
                
            # Validate batch limit
            is_valid, message = backend.analyzer.validate_batch_limit(texts, user_tier)
            
            if not is_valid:
                st.error(f"❌ {message}")
                return
            
            # Insert to batch queue
            success, result = backend.db.insert_batch_request(
                user_id="batch_user",  # For demo
                texts=texts,
                method=method,
                tier=user_tier,
                language=language
            )
            
            if success:
                st.success(f"✅ Batch analysis submitted! Queue ID: {result}")
                st.info("Click 'Process Queue' to analyze or enable auto-processing")
                
                # Show preview
                with st.expander("📋 Batch Preview"):
                    for i, text in enumerate(texts[:5]):  # Show first 5
                        st.write(f"{i+1}. {text}")
                    if len(texts) > 5:
                        st.write(f"... and {len(texts) - 5} more texts")
            else:
                st.error(f"❌ Failed to submit batch: {result}")

def show_batch_results(backend):
    """Show batch results history"""
    st.subheader("📈 Batch Results History")
    
    try:
        # Get recent batch results
        recent_batches = get_recent_batch_results(backend.db, limit=5)
        
        if recent_batches:
            for batch in recent_batches:
                with st.expander(f"📦 Batch: {batch['queue_id'][:8]}... ({batch['item_count']} texts)"):
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.write(f"**Tier:** {batch['tier']}")
                    with col2:
                        st.write(f"**Method:** {batch['method']}")
                    with col3:
                        st.write(f"**Status:** {batch['status']}")
                    with col4:
                        st.write(f"**Items:** {batch['item_count']}")
                    
                    # Show sample results
                    if batch['results']:
                        st.write("**Sample Results:**")
                        for result in batch['results'][:3]:  # Show first 3
                            sentiment_emoji = "😊" if result['sentiment_label'] == 'positive' else "😞" if result['sentiment_label'] == 'negative' else "😐"
                            st.write(f"{sentiment_emoji} `{result['text'][:50]}...` → {result['sentiment_label']} ({result['confidence_score']:.0%})")
                        
                        if len(batch['results']) > 3:
                            st.write(f"... and {len(batch['results']) - 3} more results")
        else:
            st.info("No batch analysis results yet. Submit some batch requests first!")
            
    except Exception as e:
        st.info("Batch results history coming soon...")

def get_recent_batch_results(db, limit=5):
    """Get recent batch analysis results"""
    # This would need proper implementation in database_manager
    return []

def update_process_queue_method(backend):
    """Update process_queue to handle batch items"""
    def process_queue():
        """Process both single and batch queue items"""
        try:
            queued_items = backend.db.get_queued_items(backend.batch_size)
            
            if not queued_items:
                return 0
            
            processed_count = 0
            for item in queued_items:
                try:
                    slot_id = backend.db.acquire_session_slot(item['tier'], item['user_id'])
                    if not slot_id:
                        continue
                    
                    backend.db.update_queue_status(item['queue_id'], 'processing', slot_id)
                    
                    if item.get('is_batch', False):
                        # Process batch item
                        success = backend.db.process_batch_queue_item(item['queue_id'], backend.analyzer)
                        if success:
                            processed_count += item.get('item_count', 1)
                    else:
                        # Process single item
                        result = backend.analyzer.analyze_sentiment(
                            item['input_text'], 
                            item.get('method', 'NaiveBayes'),
                            language='auto'
                        )
                        
                        backend.db.insert_result(
                            queue_id=item['queue_id'],
                            sentiment_label=result['sentiment_label'],
                            confidence_score=result['confidence_score'],
                            json_result=result,
                            processed_by=f"Streamlit_{result['method_used']}_{result['language_detected']}"
                        )
                        
                        backend.db.update_queue_status(item['queue_id'], 'done')
                        processed_count += 1
                    
                    backend.db.release_session_slot(slot_id)
                    
                except Exception as e:
                    logger.error(f"Failed to process {item['queue_id']}: {e}")
                    backend.db.update_queue_status(item['queue_id'], 'error')
            
            return processed_count
            
        except Exception as e:
            logger.error(f"Queue processing error: {e}")
            return 0
    
    return process_queue

# Update the backend class
class SentilBackend:
    def __init__(self):
        self.db = DatabaseManager()
        self.analyzer = BilingualSentimentAnalyzer()
        self.batch_size = app_config.processing_batch_size
        
        # Override process_queue method
        self.process_queue = update_process_queue_method(self)
        
        # Quick connection test
        try:
            if not self.db.test_connection():
                logger.warning("Database connection test failed")
        except:
            logger.warning("Database connection test skipped")

def main():
    """Main app function"""
    st.set_page_config(
        page_title="Sentil - Batch Analyzer", 
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 Sentil Multi-Method Sentiment Analyzer")
    st.markdown("**Single & Batch Analysis** | Tier-based Limits")
    
    # Initialize
    if not init_session_state():
        st.stop()
    
    backend = st.session_state.backend
    
    # Sidebar
    show_sidebar(backend)
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🚀 Quick Actions", 
        "📝 Single Analysis", 
        "📚 Batch Analysis", 
        "📊 Queue Status", 
        "🧪 Test Analysis"
    ])
    
    with tab1:
        show_quick_actions(backend)
    
    with tab2:
        show_input_form(backend)  # Existing single analysis form
    
    with tab3:
        show_batch_analysis_section(backend)
        show_batch_results(backend)
    
    with tab4:
        show_queue_status(backend)
        show_results_history(backend)
    
    with tab5:
        show_test_section(backend)
    
    # Footer
    st.markdown("---")
    st.caption("Sentil v1.0 | Single & Batch Analysis | Tier 1:10 texts, Tier 2:30 texts, Tier 3:100 texts")

if __name__ == "__main__":
    main()
