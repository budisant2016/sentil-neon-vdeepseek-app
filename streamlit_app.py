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
    from sentiment_analyzer import SentimentAnalyzer
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
        self.analyzer = SentimentAnalyzer()
        self.batch_size = app_config.processing_batch_size
        
        if not self.db.test_connection():
            raise Exception("Database connection failed")

    def process_queue(self):
        """Process queued items"""
        logger.info("Processing queue...")
        queued_items = self.db.get_queued_items(self.batch_size)
        
        if not queued_items:
            logger.info("No items in queue")
            return 0
        
        processed_count = 0
        for item in queued_items:
            try:
                slot_id = self.db.acquire_session_slot(item['tier'], item['user_id'])
                if not slot_id:
                    continue
                
                self.db.update_queue_status(item['queue_id'], 'processing', slot_id)
                
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
                
                logger.info(f"Processed {item['queue_id']}")
                
            except Exception as e:
                logger.error(f"Failed to process {item['queue_id']}: {e}")
                self.db.update_queue_status(item['queue_id'], 'error')
        
        return processed_count

def show_sidebar(backend):
    """Show sidebar content"""
    st.sidebar.header("Connection Info")
    
    conn_info = db_config.parse_connection_string()
    if conn_info and 'error' not in conn_info:
        st.sidebar.success("✅ DB Connected")
        st.sidebar.write(f"Host: {conn_info.get('host', 'N/A')}")
        st.sidebar.write(f"Database: {conn_info.get('database', 'N/A')}")
    
    # Test examples
    st.sidebar.subheader("Test Examples")
    examples = {
        "English": "I love this product! It's amazing and works perfectly.",
        "Indonesian": "Saya sangat suka produk ini! Kualitasnya bagus sekali.",
        "Mixed": "Produknya okay, but could be better for the price."
    }
    
    for lang, text in examples.items():
        if st.sidebar.button(f"{lang} Example"):
            st.session_state.test_text = text
            st.session_state.show_test = True
            st.rerun()

def show_test_section(backend):
    """Show test section"""
    if not st.session_state.show_test:
        return
    
    st.subheader("Test Analysis")
    
    test_text = st.text_area("Text to analyze:", st.session_state.test_text)
    col1, col2 = st.columns(2)
    
    with col1:
        method = st.selectbox("Method", ['NaiveBayes', 'KNN', 'RandomForest', 'SVM'])
    with col2:
        language = st.selectbox("Language", ['auto', 'english', 'indonesian'])
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Analyze", type="primary"):
            with st.spinner("Analyzing..."):
                try:
                    result = backend.analyzer.analyze_sentiment(test_text, method, language)
                    
                    st.success("Analysis Complete!")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Sentiment", result['sentiment_label'])
                    with col2:
                        st.metric("Confidence", f"{result['confidence_score']:.1%}")
                    with col3:
                        st.metric("Language", result['language_detected'])
                    
                    with st.expander("Details"):
                        st.json(result)
                        
                except Exception as e:
                    st.error(f"Error: {e}")
    
    with col2:
        if st.button("Close"):
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
    st.markdown("Multi-language sentiment analysis backend")
    
    # Initialize
    if not init_session_state():
        return
    
    backend = st.session_state.backend
    
    # Sidebar
    show_sidebar(backend)
    
    # Main content
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Process Queue", type="primary", use_container_width=True):
            with st.spinner("Processing..."):
                count = backend.process_queue()
                if count > 0:
                    st.success(f"Processed {count} items")
                else:
                    st.info("No items to process")
    
    with col2:
        if st.button("📊 Queue Status", use_container_width=True):
            items = backend.db.get_queued_items(5)
            st.info(f"Queued items: {len(items)}")
    
    with col3:
        if st.button("🧪 Test Analysis", use_container_width=True):
            st.session_state.show_test = True
            st.rerun()
    
    # Auto-processing
    st.subheader("Auto Processing")
    auto_process = st.checkbox("Enable auto-processing")
    
    if auto_process:
        st.info("Auto-processing enabled")
        placeholder = st.empty()
        
        stop_processing = False
        while auto_process and not stop_processing:
            with placeholder.container():
                count = backend.process_queue()
                st.write(f"Last run: {datetime.now().strftime('%H:%M:%S')}")
                st.write(f"Items processed: {count}")
                
                if st.button("Stop", key="stop_btn"):
                    stop_processing = True
                    st.rerun()
            
            time.sleep(60)  # 1 minute
    
    # Test section
    show_test_section(backend)
    
    # Footer
    st.markdown("---")
    st.caption("Sentil Backend v1.0 | Bilingual Support")

# Run the app
if __name__ == "__main__":
    main()
