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
except ImportError as e:
    st.error(f"Import error: {e}")
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

# ... (rest of the app.py remains the same as previous version)
