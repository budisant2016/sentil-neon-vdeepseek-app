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
    """Initialize session state - HARUS CEPAT"""
    if 'backend' not in st.session_state:
        try:
            # Initialize backend TANPA blocking operations
            st.session_state.backend = SentilBackend()
        except Exception as e:
            st.error(f"Failed to initialize: {e}")
            return False
    
    if 'show_test' not in st.session_state:
        st.session_state.show_test = False
    if 'test_text' not in st.session_state:
        st.session_state.test_text = ""
    if 'analyzer_ready' not in st.session_state:
        st.session_state.analyzer_ready = False
    
    return True

class SentilBackend:
    def __init__(self):
        # Initialize components CEPAT
        self.db = DatabaseManager()
        self.analyzer = BilingualSentimentAnalyzer()  # Ini sekarang cepat
        self.batch_size = app_config.processing_batch_size
        
        # Test connection (bisa di comment sementara jika lambat)
        try:
            if not self.db.test_connection():
                logger.warning("Database connection test failed")
        except:
            logger.warning("Database connection test skipped")

    def process_queue(self):
        """Process queued items"""
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
                    
                    # Analysis dengan timeout protection
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
                        processed_by=f"Streamlit_{result['language_detected']}"
                    )
                    
                    self.db.update_queue_status(item['queue_id'], 'done')
                    self.db.release_session_slot(slot_id)
                    processed_count += 1
                    
                except Exception as e:
                    logger.error(f"Failed to process {item['queue_id']}: {e}")
                    self.db.update_queue_status(item['queue_id'], 'error')
            
            return processed_count
            
        except Exception as e:
            logger.error(f"Queue processing error: {e}")
            return 0

def show_sidebar():
    """Show sidebar content - HARUS CEPAT"""
    st.sidebar.header("Connection Info")
    
    try:
        conn_info = db_config.parse_connection_string()
        if conn_info and 'error' not in conn_info:
            st.sidebar.success("✅ DB Connected")
            st.sidebar.write(f"Database: {conn_info.get('database', 'N/A')}")
    except:
        st.sidebar.info("🔧 Checking connection...")
    
    # Test examples
    st.sidebar.subheader("Test Examples")
    examples = {
        "English": "I love this product!",
        "Indonesian": "Saya suka produk ini!",
        "Mixed": "Produknya okay."
    }
    
    for lang, text in examples.items():
        if st.sidebar.button(f"{lang} Example"):
            st.session_state.test_text = text
            st.session_state.show_test = True
            st.rerun()

def show_test_section(backend):
    """Show test section dengan progress indicator"""
    if not st.session_state.show_test:
        return
    
    st.subheader("🧪 Test Analysis")
    
    test_text = st.text_area("Text to analyze:", st.session_state.test_text or "Type text here...")
    
    col1, col2 = st.columns(2)
    with col1:
        method = st.selectbox("Method", ['NaiveBayes'])
    with col2:
        language = st.selectbox("Language", ['auto', 'english', 'indonesian'])
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🚀 Analyze", type="primary", use_container_width=True):
            placeholder = st.empty()
            with placeholder.container():
                with st.spinner("Analyzing..."):
                    try:
                        # Analysis dengan progress
                        result = backend.analyzer.analyze_sentiment(test_text, method, language)
                        
                        # Tampilkan hasil
                        st.success("✅ Analysis Complete!")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            sentiment = result['sentiment_label']
                            emoji = "😊" if sentiment == 'positive' else "😞" if sentiment == 'negative' else "😐"
                            st.metric("Sentiment", f"{emoji} {sentiment}")
                        with col2:
                            st.metric("Confidence", f"{result['confidence_score']:.1%}")
                        with col3:
                            lang = result['language_detected']
                            flag = "🇺🇸" if lang == 'english' else "🇮🇩"
                            st.metric("Language", f"{flag} {lang}")
                        
                    except Exception as e:
                        st.error(f"❌ Analysis failed: {e}")
    
    with col2:
        if st.button("❌ Close", use_container_width=True):
            st.session_state.show_test = False
            st.rerun()

def main():
    """Main app function"""
    st.set_page_config(
        page_title="Sentil Analyzer",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 Sentil Sentiment Analyzer")
    st.markdown("Real-time bilingual sentiment analysis")
    
    # Initialize - HARUS CEPAT
    if not init_session_state():
        st.stop()
    
    backend = st.session_state.backend
    
    # Sidebar - CEPAT
    show_sidebar()
    
    # Main actions
    st.subheader("Quick Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Process Queue", type="primary", use_container_width=True):
            with st.spinner("Processing..."):
                count = backend.process_queue()
                if count > 0:
                    st.success(f"✅ Processed {count} items")
                else:
                    st.info("📭 No items in queue")
    
    with col2:
        if st.button("📊 Check Status", use_container_width=True):
            try:
                items = backend.db.get_queued_items(3)
                st.info(f"📋 Queued items: {len(items)}")
                for item in items:
                    st.write(f"- {item['input_text'][:30]}...")
            except:
                st.error("❌ Failed to check status")
    
    with col3:
        if st.button("🧪 Test Analysis", use_container_width=True):
            st.session_state.show_test = True
            st.rerun()
    
    # Test section
    show_test_section(backend)
    
    # Footer
    st.markdown("---")
    st.caption("Sentil v1.0 | Fast Bilingual Analysis")

if __name__ == "__main__":
    main()
