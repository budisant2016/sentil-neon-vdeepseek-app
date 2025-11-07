import streamlit as st
import time
import logging
import sys
import os
from datetime import datetime

# Fix import path untuk Streamlit Cloud
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
if src_path not in sys.path:
    sys.path.append(src_path)

try:
    from database_manager import DatabaseManager
    #from sentiment_analyzer import SentimentAnalyzer
    from sentiment_analyzer import BilingualSentimentAnalyzer
    from config import db_config, app_config
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SentilBackend:
    def __init__(self):
        self.db = DatabaseManager()
        #self.analyzer = SentimentAnalyzer()
        self.analyzer = BilingualSentimentAnalyzer()
        self.processing_batch_size = app_config.processing_batch_size
        
        # Test connection
        if not self.db.test_connection():
            raise Exception("Database connection failed")
    
    def process_queue(self):
        """Main method to process queued items"""
        logger.info("🔄 Starting queue processing...")
        
        queued_items = self.db.get_queued_items(self.processing_batch_size)
        
        if not queued_items:
            logger.info("ℹ️ No items in queue")
            return 0
        
        processed_count = 0
        
        for item in queued_items:
            try:
                logger.info(f"📥 Processing item {item['queue_id']} for user {item['user_id']}")
                
                # Acquire session slot
                slot_id = self.db.acquire_session_slot(item['tier'], item['user_id'])
                
                if not slot_id:
                    logger.warning(f"⚠️ No available slot for tier {item['tier']}")
                    continue
                
                # Update status to processing
                self.db.update_queue_status(item['queue_id'], 'processing', slot_id)
                self.db.log_system_activity('backend', f'Started processing {item["queue_id"]}', 'info', item['queue_id'])
                
                # Perform sentiment analysis dengan auto language detection
                result = self.analyzer.analyze_sentiment(
                    item['input_text'], 
                    item.get('method', 'NaiveBayes'),
                    language='auto'  # Auto-detect language
                )
                
                # Save result
                self.db.insert_result(
                    queue_id=item['queue_id'],
                    sentiment_label=result['sentiment_label'],
                    confidence_score=result['confidence_score'],
                    json_result=result,
                    processed_by=f"Streamlit_{item.get('method', 'NaiveBayes')}_{result['language_detected']}"
                )
                
                # Update queue status to done
                self.db.update_queue_status(item['queue_id'], 'done')
                self.db.release_session_slot(slot_id)
                
                self.db.log_system_activity('backend', f'Completed processing {item["queue_id"]}', 'info', item['queue_id'])
                processed_count += 1
                
                logger.info(f"✅ Successfully processed {item['queue_id']} - Language: {result['language_detected']}")
                
            except Exception as e:
                logger.error(f"❌ Failed to process {item['queue_id']}: {e}")
                self.db.update_queue_status(item['queue_id'], 'error')
                self.db.log_system_activity('backend', f'Error processing {item["queue_id"]}: {str(e)}', 'error', item['queue_id'])
        
        return processed_count

def initialize_session_state():
    """Initialize session state variables"""
    if 'backend' not in st.session_state:
        try:
            st.session_state.backend = SentilBackend()
        except Exception as e:
            st.error(f"❌ Failed to initialize backend: {e}")
            return False
    
    # Initialize other session state variables
    if 'show_test' not in st.session_state:
        st.session_state.show_test = False
    if 'auto_process_active' not in st.session_state:
        st.session_state.auto_process_active = False
    if 'test_text' not in st.session_state:
        st.session_state.test_text = ""
    
    return True

def render_sidebar(backend):
    """Render sidebar content"""
    st.sidebar.header("🔗 Connection Info")
    
    # Show connection info (without password)
    conn_info = db_config.parse_connection_string()
    if conn_info:
        if 'error' in conn_info:
            st.sidebar.error(f"Connection Error: {conn_info['error']}")
        else:
            st.sidebar.success("✅ Neon DB Connected!")
            st.sidebar.write(f"**Host:** {conn_info.get('host', 'N/A')}")
            st.sidebar.write(f"**Database:** {conn_info.get('database', 'N/A')}")
            st.sidebar.write(f"**User:** {conn_info.get('user', 'N/A')}")
            st.sidebar.write(f"**Port:** {conn_info.get('port', 'N/A')}")
            st.sidebar.write(f"**SSL:** {conn_info.get('ssl_mode', 'N/A')}")
    
    # Check if secrets are loaded
    try:
        if hasattr(st, 'secrets') and 'connections' in st.secrets:
            st.sidebar.info("🔐 Using Streamlit Secrets")
        else:
            st.sidebar.info("🔧 Using Environment Variables")
    except:
        st.sidebar.info("🔧 Using Environment Variables")
    
    # Show database info
    try:
        db_info = backend.db.get_database_info()
        if 'error' not in db_info:
            st.sidebar.subheader("📊 Database Stats")
            st.sidebar.write(f"**Tables:** {len(db_info.get('tables', []))} tables")
            queue_stats = db_info.get('queue_stats', {})
            if queue_stats:
                st.sidebar.write("**Queue Status:**")
                for status, count in queue_stats.items():
                    st.sidebar.write(f"  - {status}: {count}")
    except Exception as e:
        st.sidebar.error(f"Database info error: {e}")
    
    # Test examples
    st.sidebar.subheader("🌐 Test Examples")
    if st.sidebar.button("English Example"):
        st.session_state.test_text = "I absolutely love this product! It's fantastic and works perfectly."
        st.session_state.show_test = True
        st.rerun()

    if st.sidebar.button("Indonesian Example"):
        st.session_state.test_text = "Saya sangat suka produk ini! Kualitasnya bagus dan harganya terjangkau."
        st.session_state.show_test = True
        st.rerun()

    if st.sidebar.button("Mixed Example"):
        st.session_state.test_text = "Produknya okay, but could be better. Cukup lumayan untuk harganya."
        st.session_state.show_test = True
        st.rerun()

def render_test_section(backend):
    """Render manual test section"""
    if st.session_state.show_test:
        st.subheader("🧪 Manual Test Analysis")
        
        col1, col2 = st.columns(2)
        with col1:
            test_text = st.text_area("Test text:", st.session_state.test_text or "I love this product! It's amazing!")
        with col2:
            test_method = st.selectbox("Method", ['NaiveBayes', 'KNN', 'RandomForest', 'SVM'])
            test_language = st.selectbox("Language", ['auto', 'english', 'indonesian'], 
                                       help="Auto: detect automatically, English/Indonesian: force specific language")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Run Test", type="primary"):
                with st.spinner("Analyzing..."):
                    try:
                        result = backend.analyzer.analyze_sentiment(test_text, test_method, test_language)
                        
                        st.subheader("📊 Results")
                        
                        # Display main results
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            sentiment_color = "🟢" if result['sentiment_label'] == 'positive' else "🔴" if result['sentiment_label'] == 'negative' else "🟡"
                            st.metric("Sentiment", f"{sentiment_color} {result['sentiment_label']}")
                        with col2:
                            st.metric("Confidence", f"{result['confidence_score']:.2%}")
                        with col3:
                            st.metric("Method", result['method_used'])
                        with col4:
                            lang_flag = "🇺🇸" if result['language_detected'] == 'english' else "🇮🇩"
                            st.metric("Language", f"{lang_flag} {result['language_detected']}")
                        
                        # Additional info
                        with st.expander("Detailed Results"):
                            st.json(result)
                            
                        # Language detection info
                        st.info(f"**Language Detection:** {result['language_detected'].upper()} - {result['processed_text']}")
                            
                    except Exception as e:
                        st.error(f"Test failed: {e}")
        
        with col2:
            if st.button("Close Test"):
                st.session_state.show_test = False
                st.rerun()

def render_auto_processing(backend):
    """Render auto-processing section"""
    st.subheader("⚙️ Auto-Processing")
    
    col1, col2 = st.columns(2)
    with col1:
        auto_process = st.checkbox("Enable Auto-processing", value=False)
    with col2:
        processing_interval = st.slider("Interval (seconds)", 10, 300, 60)
    
    if auto_process:
        st.info("🔄 Auto-processing is enabled")
        placeholder = st.empty()
        
        # Initialize auto-process state
        st.session_state.auto_process_active = True
        
        while auto_process and st.session_state.auto_process_active:
            with placeholder.container():
                try:
                    processed = backend.process_queue()
                    current_time = datetime.now().strftime('%H:%M:%S')
                    
                    st.write(f"**Last run:** {current_time}")
                    st.write(f"**Items processed:** {processed}")
                    
                    if processed == 0:
                        st.write("Queue is empty, waiting...")
                    
                    if st.button("🛑 Stop Auto-Process"):
                        st.session_state.auto_process_active = False
                        st.rerun()
                        
                except Exception as e:
                    st.error(f"Auto-processing error: {e}")
                    break
            
            time.sleep(processing_interval)

def main():
    st.set_page_config(
        page_title="Sentil Backend Processor",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 Sentil Backend Processor")
    st.markdown("Real-time sentiment analysis backend service - **Streamlit Cloud + Neon DB**")
    
    # Initialize session state
    if not initialize_session_state():
        return
    
    backend = st.session_state.backend
    
    # Render sidebar
    render_sidebar(backend)
    
    # Quick actions
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Process Queue Now", type="primary", use_container_width=True):
            with st.spinner("Processing queue..."):
                processed = backend.process_queue()
                if processed > 0:
                    st.success(f"✅ Processed {processed} items")
                else:
                    st.info("ℹ️ No items to process")
    
    with col2:
        if st.button("📊 Refresh Stats", use_container_width=True):
            st.rerun()
    
    with col3:
        if st.button("🧪 Test Analysis", use_container_width=True):
            st.session_state.show_test = True
            st.rerun()
    
    # Render test section
    render_test_section(backend)
    
    # Render auto-processing section
    render_auto_processing(backend)
    
    # Footer
    st.markdown("---")
    st.caption("Sentil Backend v1.0 - Streamlit Cloud + Neon DB")

if __name__ == "__main__":
    main()
