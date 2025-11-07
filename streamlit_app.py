import streamlit as st
import time
import logging
import sys
import os
from datetime import datetime

# Add src to Python path untuk Streamlit Cloud
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.database_manager import DatabaseManager
    from src.sentiment_analyzer import SentimentAnalyzer
    from src.config import app_config
except ImportError:
    # Fallback untuk development lokal
    from database_manager import DatabaseManager
    from sentiment_analyzer import SentimentAnalyzer
    from config import app_config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SentilBackend:
    def __init__(self):
        self.db = DatabaseManager()
        self.analyzer = SentimentAnalyzer()
        self.processing_batch_size = app_config.processing_batch_size
    
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
                
                # Perform sentiment analysis
                result = self.analyzer.analyze_sentiment(
                    item['input_text'], 
                    item['method']
                )
                
                # Save result
                self.db.insert_result(
                    queue_id=item['queue_id'],
                    sentiment_label=result['sentiment_label'],
                    confidence_score=result['confidence_score'],
                    json_result=result,
                    processed_by=f"Streamlit_{item['method']}"
                )
                
                # Update queue status to done
                self.db.update_queue_status(item['queue_id'], 'done')
                self.db.release_session_slot(slot_id)
                
                self.db.log_system_activity('backend', f'Completed processing {item["queue_id"]}', 'info', item['queue_id'])
                processed_count += 1
                
                logger.info(f"✅ Successfully processed {item['queue_id']}")
                
            except Exception as e:
                logger.error(f"❌ Failed to process {item['queue_id']}: {e}")
                self.db.update_queue_status(item['queue_id'], 'error')
                self.db.log_system_activity('backend', f'Error processing {item["queue_id"]}: {str(e)}', 'error', item['queue_id'])
        
        return processed_count

def main():
    st.set_page_config(
        page_title="Sentil Backend Processor",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 Sentil Backend Processor")
    st.markdown("Real-time sentiment analysis backend service - **Streamlit Cloud Edition**")
    
    # Initialize backend
    if 'backend' not in st.session_state:
        try:
            st.session_state.backend = SentilBackend()
            st.success("✅ Backend initialized successfully!")
        except Exception as e:
            st.error(f"❌ Failed to initialize backend: {e}")
            return
    
    backend = st.session_state.backend
    
    # Sidebar for controls
    st.sidebar.header("Control Panel")
    
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
        if st.button("📊 Queue Status", use_container_width=True):
            queued_items = backend.db.get_queued_items(10)
            st.info(f"Items in queue: {len(queued_items)}")
            
            if queued_items:
                for item in queued_items:
                    st.write(f"- `{item['queue_id'][:8]}...` (Tier {item['tier']}) - {item['method']}")
    
    with col3:
        if st.button("🧪 Test Analysis", use_container_width=True):
            st.session_state.show_test = True
    
    # Statistics
    st.sidebar.subheader("System Stats")
    try:
        stats = backend.db.get_queue_stats()
        if stats:
            for stat in stats:
                st.sidebar.write(f"Tier {stat['tier']} - {stat['status']}: {stat['count']}")
        else:
            st.sidebar.info("No queue statistics available")
    except Exception as e:
        st.sidebar.error(f"Stats error: {e}")
    
    # Manual test section
    if st.session_state.get('show_test', False):
        st.subheader("🧪 Manual Test Analysis")
        test_text = st.text_area("Test text for sentiment analysis:", 
                                "I love this product! It's amazing!")
        test_method = st.selectbox("Method", ['NaiveBayes', 'KNN', 'RandomForest', 'SVM'])
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Run Test", type="primary"):
                with st.spinner("Analyzing..."):
                    try:
                        result = backend.analyzer.analyze_sentiment(test_text, test_method)
                        st.json(result)
                        
                        # Display results nicely
                        st.subheader("📊 Results")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Sentiment", result['sentiment_label'])
                        with col2:
                            st.metric("Confidence", f"{result['confidence_score']:.2%}")
                        with col3:
                            st.metric("Method", result['method_used'])
                            
                    except Exception as e:
                        st.error(f"Test failed: {e}")
        
        with col2:
            if st.button("Close Test"):
                st.session_state.show_test = False
                st.rerun()
    
    # Auto-processing section
    st.subheader("⚙️ Auto-Processing Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        auto_process = st.checkbox("Enable Auto-processing", value=False, 
                                  help="Automatically process queue at set intervals")
    
    with col2:
        processing_interval = st.slider("Processing Interval (seconds)", 
                                      10, 300, 60, 
                                      help="How often to check for new queue items")
    
    # Auto-processing loop
    if auto_process:
        st.info("🔄 Auto-processing is enabled")
        
        # Initialize auto-process state
        if 'auto_process_active' not in st.session_state:
            st.session_state.auto_process_active = True
        
        # Create placeholder for live updates
        placeholder = st.empty()
        
        # Processing loop
        while (auto_process and 
               st.session_state.get('auto_process_active', True) and 
               st.session_state.get('auto_process', True)):
            
            with placeholder.container():
                try:
                    processed = backend.process_queue()
                    
                    st.write(f"**Last run:** {datetime.now().strftime('%H:%M:%S')}")
                    st.write(f"**Items processed:** {processed}")
                    
                    if processed == 0:
                        st.write("Queue is empty, waiting...")
                    
                    # Add stop button inside the loop
                    if st.button("🛑 Stop Auto-Process", key="stop_auto"):
                        st.session_state.auto_process_active = False
                        st.session_state.auto_process = False
                        st.rerun()
                        
                except Exception as e:
                    st.error(f"Auto-processing error: {e}")
                    break
            
            # Wait for next interval
            time.sleep(processing_interval)
    
    # Footer
    st.markdown("---")
    st.caption("Sentil Backend v1.0 - Streamlit Cloud Deployment")

if __name__ == "__main__":
    main()
