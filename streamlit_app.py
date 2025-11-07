import streamlit as st
import time
import logging
from datetime import datetime
from database_manager import DatabaseManager
from sentiment_analyzer import SentimentAnalyzer

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
        self.processing_batch_size = 5
    
    def process_queue(self):
        """Main method to process queued items"""
        logger.info("Starting queue processing...")
        
        # Get queued items
        queued_items = self.db.get_queued_items(self.processing_batch_size)
        
        if not queued_items:
            logger.info("No items in queue")
            return 0
        
        processed_count = 0
        
        for item in queued_items:
            try:
                logger.info(f"Processing item {item['queue_id']} for user {item['user_id']}")
                
                # Acquire session slot
                slot_id = self.db.acquire_session_slot(item['tier'], item['user_id'])
                
                if not slot_id:
                    logger.warning(f"No available slot for tier {item['tier']}")
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
                
                # Release session slot
                self.db.release_session_slot(slot_id)
                
                self.db.log_system_activity('backend', f'Completed processing {item["queue_id"]}', 'info', item['queue_id'])
                processed_count += 1
                
                logger.info(f"Successfully processed {item['queue_id']}")
                
            except Exception as e:
                logger.error(f"Failed to process {item['queue_id']}: {e}")
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
    st.markdown("Real-time sentiment analysis backend service")
    
    backend = SentilBackend()
    
    # Control panel
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Process Queue Now"):
            with st.spinner("Processing queue..."):
                processed = backend.process_queue()
                st.success(f"Processed {processed} items")
    
    with col2:
        if st.button("📊 Show Queue Status"):
            queued_items = backend.db.get_queued_items(10)
            st.info(f"Items in queue: {len(queued_items)}")
            
            for item in queued_items:
                st.write(f"- {item['queue_id'][:8]}... (Tier {item['tier']})")
    
    with col3:
        if st.button("🧹 Clear Completed"):
            # This would be implemented to clean old records
            st.info("Cleanup feature to be implemented")
    
    # Auto-processing toggle
    auto_process = st.checkbox("Enable Auto-processing", value=False)
    processing_interval = st.slider("Processing Interval (seconds)", 10, 300, 60)
    
    if auto_process:
        st.info("Auto-processing is enabled")
        placeholder = st.empty()
        
        while auto_process:
            with placeholder.container():
                processed = backend.process_queue()
                st.write(f"Last run: {datetime.now().strftime('%H:%M:%S')}")
                st.write(f"Items processed: {processed}")
                
                if processed == 0:
                    st.write("Queue is empty, waiting...")
            
            time.sleep(processing_interval)
            
            # Check if user disabled auto-processing
            if not st.session_state.get('auto_process', True):
                break

if __name__ == "__main__":
    main()
